# -*- coding: utf-8 -*-
"""
check_val_gt.py
用途：
1) 严格检查 COCO val.json 的 GT 框是否越界、为负、格式错误（应为 [x,y,w,h] 像素坐标）。
2) 检查 images[*].width/height 与真实图片尺寸是否一致、图片是否存在。
3) 统计可疑“归一化 bbox（<=1）”或“误写成 [x1,y1,x2,y2]”的比例。
4) 可选：把前 N 张图片的 GT 画出来保存到目录，便于肉眼验证。

依赖：Python 3、Pillow
pip install pillow
"""

import os
import json
import argparse
from collections import defaultdict, Counter
from PIL import Image, ImageDraw

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO 标注文件（如 data/drone_sim_coco/val.json）")
    ap.add_argument("--img-dir", required=True, help="图片目录（等价于 data_prefix.img）")
    ap.add_argument("--save-vis", default="", help="若指定，将把前 --max-vis 张图画上 GT 保存到该目录")
    ap.add_argument("--max-vis", type=int, default=50, help="可视化的图片数量上限")
    ap.add_argument("--check-pixels", type=int, default=50, help="抽样用 PIL 实测图片尺寸的张数（0 关闭）")
    ap.add_argument("--warn-topk", type=int, default=10, help="打印前多少条可疑样本的详情")
    return ap.parse_args()

def resolve_img_path(img_dir, file_name):
    """
    解析图片路径的策略：
      1) 若 file_name 是绝对路径且存在，直接用；
      2) 尝试 os.path.join(img_dir, file_name)；
      3) 尝试 os.path.join(img_dir, basename(file_name))（防止 file_name 自带子目录导致重复）。
    """
    candidates = []
    if os.path.isabs(file_name):
        candidates.append(file_name)
    candidates.append(os.path.join(img_dir, file_name))
    candidates.append(os.path.join(img_dir, os.path.basename(file_name)))
    for p in candidates:
        if os.path.exists(p):
            return p, candidates
    return None, candidates

def draw_gts(img_path, anns, out_path):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    for a in anns:
        x, y, w, h = a["bbox"]
        # 画框（红色、不透明）
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path)

def main():
    args = parse_args()

    with open(args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    print(f"共 {len(images)} 张图片，{len(anns)} 条标注，{len(cats)} 个类别。")

    id2img = {im["id"]: im for im in images}
    id2cat = {c["id"]: c for c in cats}

    # 统计
    missing_files = []
    size_mismatch = []  # json 宽高与真实图片不一致
    illegal_boxes = []  # 越界/负数/零面积
    probably_normalized = 0  # bbox 可能归一化到 0~1
    probably_xyxy = 0        # bbox 可能误写成 [x1,y1,x2,y2]
    per_image_anns = defaultdict(list)
    path_resolve_counter = Counter()

    # 先检查文件是否存在，并建立 image_id -> anns
    for a in anns:
        img_info = id2img.get(a["image_id"])
        if img_info is None:
            continue
        per_image_anns[a["image_id"]].append(a)

    # 遍历图片做检查
    confirmed_pixels = 0
    vis_saved = 0

    for idx, (img_id, img_info) in enumerate(id2img.items(), start=1):
        fname = img_info.get("file_name", "")
        W_json = img_info.get("width", None)
        H_json = img_info.get("height", None)

        img_path, tried = resolve_img_path(args.img_dir, fname)
        if img_path is None:
            missing_files.append((img_id, fname, tried[:2]))
            continue
        else:
            # 统计路径解析命中类型
            if os.path.isabs(fname) and img_path == fname:
                path_resolve_counter["abs"] += 1
            elif img_path == os.path.join(args.img_dir, fname):
                path_resolve_counter["join_dir+file_name"] += 1
            else:
                path_resolve_counter["join_dir+basename"] += 1

        # 可选：抽样用 PIL 读取真实尺寸，核对 json 宽高
        W_real = H_real = None
        if args.check_pixels > 0 and confirmed_pixels < args.check_pixels:
            try:
                with Image.open(img_path) as tmp:
                    W_real, H_real = tmp.size
                confirmed_pixels += 1
                if W_json is not None and H_json is not None:
                    if abs(W_real - W_json) > 1 or abs(H_real - H_json) > 1:
                        size_mismatch.append((img_id, fname, (W_json, H_json), (W_real, H_real)))
            except Exception as e:
                print(f"[WARN] 打开图片失败：{img_path} | {e}")

        # 检查 bbox 合法性
        W = W_json if W_json is not None else W_real
        H = H_json if H_json is not None else H_real

        anns_img = per_image_anns.get(img_id, [])
        for a in anns_img:
            bbox = a.get("bbox", None)
            if not bbox or len(bbox) != 4:
                illegal_boxes.append((img_id, fname, "bbox_missing_or_len!=4", a.get("id")))
                continue
            x, y, w, h = bbox

            # 归一化的启发式：四个量都在 [0,1] 内，且图片尺寸看起来 > 1（如果已知）
            if W and H:
                if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                    probably_normalized += 1

            # 合法性：非负 + 宽高 > 0 + 不越界（允许 1px 容差）
            tol = 1.0
            bad_reason = None
            if w <= 0 or h <= 0:
                bad_reason = "non_positive_wh"
            if x < -tol or y < -tol:
                bad_reason = "negative_xy"
            if W and (x + w > W + tol):
                bad_reason = f"x+w>{W}"
            if H and (y + h > H + tol):
                bad_reason = f"y+h>{H}"

            if bad_reason:
                illegal_boxes.append((img_id, fname, bad_reason, a.get("id")))

            # 误写成 [x1,y1,x2,y2] 的启发式：w/h 很大且 x+w 或 y+h 远超 W/H，但 w/h 本身小于 W/H
            if W and H:
                if (x + w > W + 5 or y + h > H + 5) and (0 < w < W and 0 < h < H):
                    probably_xyxy += 1

        # 可视化
        if args.save_vis and vis_saved < args.max_vis and anns_img:
            out_dir = args.save_vis
            out_path = os.path.join(out_dir, os.path.basename(img_path))
            try:
                draw_gts(img_path, anns_img, out_path)
                vis_saved += 1
            except Exception as e:
                print(f"[WARN] 可视化失败：{img_path} | {e}")

    # 汇总结果
    print("\n==== 检查结果汇总 ====")
    print(f"图片缺失：{len(missing_files)}")
    print(f"宽高不一致：{len(size_mismatch)}  (抽样检查 {confirmed_pixels} 张)")
    print(f"非法/越界 bbox：{len(illegal_boxes)}")
    print(f"疑似归一化 bbox：{probably_normalized}")
    print(f"疑似 [x1,y1,x2,y2] 误写：{probably_xyxy}")
    print("路径解析命中统计：", dict(path_resolve_counter))

    # 打印部分详情
    def head(items, k):
        return items[:min(len(items), k)]

    if missing_files:
        print("\n[样例] 缺失图片（前 {} 条）：".format(args.warn_topk))
        for img_id, fname, tried in head(missing_files, args.warn_topk):
            print(f" img_id={img_id} file_name='{fname}'  尝试路径={tried}")

    if size_mismatch:
        print("\n[样例] 宽高不一致（前 {} 条）：".format(args.warn_topk))
        for img_id, fname, wh_json, wh_real in head(size_mismatch, args.warn_topk):
            print(f" img_id={img_id} file_name='{fname}'  json={wh_json}  real={wh_real}")

    if illegal_boxes:
        print("\n[样例] 非法/越界 bbox（前 {} 条）：".format(args.warn_topk))
        for img_id, fname, reason, ann_id in head(illegal_boxes, args.warn_topk):
            print(f" img_id={img_id} ann_id={ann_id} file_name='{fname}'  reason={reason}")

    if args.save_vis:
        print(f"\n已保存可视化 GT：{vis_saved} 张 -> {args.save_vis}")
        if vis_saved == 0:
            print("提示：没有保存到任何图片，可能是 --max-vis=0 或该 val.json 没有标注。")

    print("\n结论判定建议：")
    print(" - 若 missing_files>0：data_prefix.img 与 file_name 不匹配，或路径写错。")
    print(" - 若 probably_normalized 比例很高：bbox 可能被错误归一化成了 0~1。")
    print(" - 若 probably_xyxy 明显>0：bbox 可能误写为 [x1,y1,x2,y2]。")
    print(" - 若 size_mismatch>0：images[*].width/height 与真实图片不符，应修正 json。")
    print(" - 若 illegal_boxes>0：存在越界/负值/零面积 bbox，需修正。")

if __name__ == "__main__":
    main()
