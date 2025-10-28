#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anti-UAV -> COCO 转换脚本
- 支持 modality: visible / infrared / both
- 默认导出带标注的 COCO（仅在 exist==1 时写入 bbox）
- 可选同时导出“无标注版”COCO（仅含 images，无 annotations），便于做 UDA 目标域
- COCO 类别固定为 1 类：drone
使用示例：
  python tools/convert/anti_uav_to_coco.py \
      --root data/anti_uav \
      --out  data/anti_uav_coco \
      --modality visible \
      --include-empty \
      --export-unlabeled
Anti-UAV -> COCO 转换脚本（含可视化抽样）
- 支持 modality: visible / infrared / both
- 默认导出带标注的 COCO（仅在 exist==1 时写入 bbox）
- 可选同时导出“无标注版”COCO（仅含 images，无 annotations），便于做 UDA 目标域
- 新增可视化：
    --vis-out <dir>       开启并指定输出目录
    --vis-max <N>         每个 split 抽 N 张「有框」图（默认 50）
    --vis-empty <M>       额外抽 M 张「空帧」图（默认 5）
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


CATEGORIES = [{"id": 1, "name": "drone", "supercategory": "drone"}]


def natural_key(p: Path):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(p))]


def read_seq_json(seq_dir: Path, modality: str) -> Dict[str, Any]:
    jpath = seq_dir / f"{modality}.json"
    if not jpath.exists():
        raise FileNotFoundError(f"标注缺失：{jpath}")
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    exist = data.get("exist") or data.get("Exist") or data.get("exists")
    gt_rect = data.get("gt_rect") or data.get("gtRect") or data.get("gt_bbox")
    if exist is None or gt_rect is None:
        raise KeyError(f"{jpath} 缺少 `exist` 或 `gt_rect` 字段")
    return {"exist": exist, "gt_rect": gt_rect}


def frame_boxes(exist_list: List[int], rect_list: List[Any], idx: int) -> List[List[float]]:
    if idx >= len(exist_list) or idx >= len(rect_list):
        return []
    if not exist_list[idx]:
        return []
    rect = rect_list[idx]
    if isinstance(rect, (list, tuple)) and rect and isinstance(rect[0], (int, float)):
        x, y, w, h = rect[:4]
        if w > 0 and h > 0:
            return [[float(x), float(y), float(w), float(h)]]
        return []
    if isinstance(rect, (list, tuple)) and rect and isinstance(rect[0], (list, tuple)):
        out = []
        for r in rect:
            if len(r) >= 4 and r[2] > 0 and r[3] > 0:
                out.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        return out
    return []


def collect_images(seq_dir: Path, modality: str) -> List[Path]:
    img_dir = seq_dir / modality
    if not img_dir.is_dir():
        raise FileNotFoundError(f"图片目录缺失：{img_dir}")
    imgs = sorted(list(img_dir.glob(f"{modality}*.jpg")), key=natural_key)
    if not imgs:
        imgs = sorted(list(img_dir.glob("*.jpg")), key=natural_key)
    if not imgs:
        raise FileNotFoundError(f"未发现图像：{img_dir}")
    return imgs


def get_image_hw(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        w, h = im.size
    return h, w


def draw_boxes(img_path: Path, boxes: List[List[float]], save_path: Path,
               title: str = "drone") -> None:
    """在图像上画框并保存。"""
    with Image.open(img_path) as im:
        im = im.convert("RGB")  # 统一到 RGB，IR 单通道也能画色框
        draw = ImageDraw.Draw(im)
        # 线宽按短边比例自适应
        lw = max(2, min(im.size) // 300)
        for b in boxes:
            x, y, w, h = b
            x2, y2 = x + w, y + h
            draw.rectangle([(x, y), (x2, y2)], outline=(255, 0, 0), width=lw)
            # 画标签背景条
            tag = f"{title}"
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            if hasattr(draw, "textbbox"):
                l, t, r, b = draw.textbbox((0, 0), tag, font=font)
                tw, th = r - l, b - t
            else:
                try:
                    tw, th = draw.textsize(tag, font=font)  # 兼容旧 Pillow
                except Exception:
                    try:
                        l, t, r, b = font.getbbox(tag)
                        tw, th = r - l, b - t
                    except Exception:
                        tw, th = 8 * len(tag), 12  # 最差兜底
            pad = max(1, lw)
            bx2 = min(x + tw + 2 * pad, im.size[0] - 1)
            by2 = y + th + 2 * pad
            draw.rectangle([(x, y), (bx2, by2)], fill=(255, 0, 0))
            draw.text((x + pad, y + pad), tag, fill=(255, 255, 255), font=font)
        im.save(save_path, quality=90)


def convert_split(
    root: Path,
    out_dir: Path,
    split: str,
    modality: str,
    include_empty: bool = True,
    export_unlabeled: bool = False,
    vis_dir: Path | None = None,
    vis_max: int = 50,
    vis_empty: int = 5,
) -> None:
    split_dir = root / split
    if not split_dir.is_dir():
        print(f"[跳过] 未找到 split 目录：{split_dir}")
        return

    coco_images: List[Dict[str, Any]] = []
    coco_anns: List[Dict[str, Any]] = []
    categories = CATEGORIES.copy()
    img_id = 1
    ann_id = 1

    # 可视化计数
    vis_has_box_count = 0
    vis_empty_count = 0
    if vis_dir is not None:
        vis_dir = vis_dir / f"{split}_{modality}"
        vis_dir.mkdir(parents=True, exist_ok=True)

    seq_list = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=natural_key)
    for seq in tqdm(seq_list, desc=f"[{split}-{modality}] 序列处理中", ncols=100):
        try:
            seq_meta = read_seq_json(seq, modality)
            exist = seq_meta["exist"]
            gt_rect = seq_meta["gt_rect"]
        except Exception as e:
            print(f"[警告] 读取 {seq} 失败：{e}")
            continue

        frames = collect_images(seq, modality)
        if len(frames) != len(exist):
            n = min(len(frames), len(exist), len(gt_rect))
            print(f"[提示] {seq.name} 帧数与标注长度不一致：frames={len(frames)}, exist={len(exist)}, gt_rect={len(gt_rect)} -> 对齐到 {n}")
            frames = frames[:n]
            exist = exist[:n]
            gt_rect = gt_rect[:n]

        H, W = get_image_hw(frames[0])

        for i, img_path in enumerate(frames):
            rel_file = img_path.relative_to(root)
            image_rec = {
                "id": img_id,
                "file_name": str(rel_file).replace("\\", "/"),
                "height": H,
                "width": W,
            }

            boxes = frame_boxes(exist, gt_rect, i)

            if boxes or include_empty:
                coco_images.append(image_rec)

            for box in boxes:
                x, y, w, h = box
                coco_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(max(w, 0.0) * max(h, 0.0)),
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1

            # 可视化抽样：优先抽有框图；不足时再抽空帧
            if vis_dir is not None:
                save_name = f"{seq.name}_{img_path.stem}.jpg"
                save_path = vis_dir / save_name
                if boxes and vis_has_box_count < vis_max:
                    try:
                        draw_boxes(img_path, boxes, save_path, title="drone")
                        vis_has_box_count += 1
                    except Exception as e:
                        print(f"[可视化失败] {img_path}: {e}")
                elif (not boxes) and include_empty and vis_empty_count < vis_empty:
                    # 空帧也导出一小部分，便于核验 exist==0 情况
                    try:
                        # 空帧写个“NO GT”角标（不画框）
                        with Image.open(img_path) as im:
                            im = im.convert("RGB")
                            draw = ImageDraw.Draw(im)
                            tag = "NO GT"
                            try:
                                font = ImageFont.load_default()
                            except Exception:
                                font = None
                            if hasattr(draw, "textbbox"):
                                l, t, r, b = draw.textbbox((0, 0), tag, font=font)
                                tw, th = r - l, b - t
                            else:
                                try:
                                    tw, th = draw.textsize(tag, font=font)  # 兼容旧 Pillow
                                except Exception:
                                    try:
                                        l, t, r, b = font.getbbox(tag)
                                        tw, th = r - l, b - t
                                    except Exception:
                                        tw, th = 8 * len(tag), 12  # 最差兜底
                            pad = 4
                            draw.rectangle([(0, 0), (tw + 2 * pad, th + 2 * pad)], fill=(0, 0, 0))
                            draw.text((pad, pad), tag, fill=(255, 255, 255), font=font)
                            im.save(save_path, quality=90)
                        vis_empty_count += 1
                    except Exception as e:
                        print(f"[可视化失败] {img_path}: {e}")

            img_id += 1

    out_json = out_dir / f"{split}_{modality}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "info": {"version": "anti-uav-1.0", "description": f"{split}/{modality} converted to COCO"},
            "licenses": [],
            "images": coco_images,
            "annotations": coco_anns,
            "categories": categories
        }, f, ensure_ascii=False)

    print(f"[完成] {split}-{modality} 带标注 COCO -> {out_json}  (images={len(coco_images)}, anns={len(coco_anns)})")

    if out_dir and vis_dir is not None:
        print(f"[可视化] 已导出到 {vis_dir}  (有框 {vis_has_box_count} 张, 空帧 {vis_empty_count} 张)")

    if export_unlabeled:
        out_json_u = out_dir / f"{split}_{modality}_unlabeled.json"
        with open(out_json_u, "w", encoding="utf-8") as f:
            json.dump({
                "info": {"version": "anti-uav-1.0", "description": f"{split}/{modality} images only (unlabeled)"},
                "licenses": [],
                "images": coco_images,
                "annotations": [],
                "categories": CATEGORIES
            }, f, ensure_ascii=False)
        print(f"[完成] {split}-{modality} 无标注 COCO -> {out_json_u}  (images={len(coco_images)})")

def _create_symlink_tree(src_root: Path, dst_root: Path, splits: List[str], modality: str) -> None:
    """
    将 src_root 下指定 splits 的 <split>/<seq>/<modality>/*.jpg 软链到 dst_root，
    并完全保留其子目录结构，以匹配 COCO JSON 的 file_name。
    """
    print(f"[symlink] {modality} -> {dst_root}")
    for split in splits:
        split_dir = src_root / split
        if not split_dir.is_dir():
            print(f"[symlink][跳过] 未找到 split 目录：{split_dir}")
            continue
        seq_list = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=natural_key)
        for seq in tqdm(seq_list, desc=f"[symlink {split}-{modality}]", ncols=100):
            img_dir = seq / modality
            if not img_dir.is_dir():
                continue
            for img_path in sorted(img_dir.glob("*.jpg"), key=natural_key):
                # 目标链接路径：dst_root/<split>/<seq.name>/<modality>/<filename>
                rel_sub = Path(split) / seq.name / modality
                link_dir = dst_root / rel_sub
                link_dir.mkdir(parents=True, exist_ok=True)
                link_path = link_dir / img_path.name

                # 已存在则跳过（如需强制重建可先 unlink）
                if link_path.exists():
                    continue

                # 使用相对路径创建软链，便于项目内移动
                try:
                    rel_target = os.path.relpath(img_path, start=link_dir)
                    link_path.symlink_to(rel_target)
                except FileExistsError:
                    pass
                except Exception as e:
                    print(f"[symlink][失败] {link_path} -> {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Anti-UAV 数据根目录（包含 train/val/test 子目录）")
    parser.add_argument("--out", type=str, required=True,
                        help="输出目录，保存 COCO JSON")
    parser.add_argument("--modality", type=str, default="visible",
                        choices=["visible", "infrared", "both"],
                        help="转换哪种模态")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="需要转换的划分，逗号分隔（train,val,test 子集均可）")
    parser.add_argument("--include-empty", action="store_true",
                        help="是否把 exist==0 的空帧也写入 images（annotations 为空）")
    parser.add_argument("--export-unlabeled", action="store_true",
                        help="同时导出无标注版 COCO（仅 images），便于作为 UDA 目标域")

    # 可视化参数
    parser.add_argument("--vis-out", type=str, default=None,
                        help="若提供，则导出抽样可视化到该目录（示例: data/anti_uav_coco/vis）")
    parser.add_argument("--vis-max", type=int, default=50,
                        help="每个 split 抽取的『有框』可视化张数上限")
    parser.add_argument("--vis-empty", type=int, default=5,
                        
                        help="每个 split 抽取的『空帧』可视化张数上限（需 --include-empty）")
    # 软链接参数
    parser.add_argument("--make-symlinks", action="store_true",
                        help="将图片软链接到 data/real_drone_rgb 和 data/real_drone_ir（或自定义路径）")
    parser.add_argument("--rgb-link-root", type=str, default="/mnt/ssd/lql/Fitness-Generalization-Transferability/data/real_drone_rgb",
                        help="RGB(visible) 软链接根目录（默认 data/real_drone_rgb）")
    parser.add_argument("--ir-link-root", type=str, default="/mnt/ssd/lql/Fitness-Generalization-Transferability/data/real_drone_ir",
                        help="IR(infrared) 软链接根目录（默认 data/real_drone_ir）")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not root.exists():
        print(f"[错误] root 不存在：{root}")
        sys.exit(1)

    modalities = ["visible", "infrared"] if args.modality == "both" else [args.modality]
    vis_dir = Path(args.vis_out).resolve() if args.vis_out else None

    for m in modalities:
        for sp in splits:
            convert_split(
                root, out_dir, sp, m,
                include_empty=args.include_empty,
                export_unlabeled=args.export_unlabeled,
                vis_dir=vis_dir,
                vis_max=args.vis_max,
                vis_empty=args.vis_empty
            )

    # ===== 在所有 JSON/可视化完成后，再做软链接 =====
    if args.make_symlinks:
        if "visible" in modalities:
            _create_symlink_tree(root, Path(args.rgb_link_root).resolve(), splits, "visible")
        if "infrared" in modalities:
            _create_symlink_tree(root, Path(args.ir_link_root).resolve(), splits, "infrared")
        print("[symlink] 完成。")

if __name__ == "__main__":
    main()
