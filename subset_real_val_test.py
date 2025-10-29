#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将真实数据的 val/test COCO 标注随机抽取一定比例（默认 10%）。
- 支持模态选择：visible(=RGB), infrared(=IR)
- 仅改写 images/annotations 列表及相关 id，保持 categories/licences/info 基本不变
- 结果文件命名：{split}_{modality}_{XX}pct.json
"""
import os, os.path as osp, json, random, argparse

def subset_coco(in_path: str, out_path: str, ratio: float, seed: int):
    with open(in_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    anns = data.get('annotations', [])
    n_total = len(images)
    if n_total == 0:
        print(f"[WARN] {in_path} 无 images，跳过。")
        return

    k = max(1, int(round(n_total * ratio)))
    rng = random.Random(seed)

    # 为了结果可复现：先按 (file_name, 原id) 排序，再采样索引
    order = list(range(n_total))
    order.sort(key=lambda i: (images[i].get('file_name',''), images[i].get('id', 0)))
    chosen = sorted(rng.sample(order, k))

    chosen_img_ids_orig = {images[i]['id'] for i in chosen}
    new_images = [images[i].copy() for i in chosen]
    new_anns = [a.copy() for a in anns if a.get('image_id') in chosen_img_ids_orig]

    # 重新编号，确保 COCO id 连续从1开始
    id_map = {img['id']: i+1 for i, img in enumerate(new_images)}
    for i, img in enumerate(new_images):
        img['id'] = i+1
    for j, a in enumerate(new_anns):
        a['id'] = j+1
        a['image_id'] = id_map[a['image_id']]

    out = {
        "info": {**data.get("info", {}), "subset_ratio": ratio, "source_file": osp.basename(in_path)},
        "licenses": data.get("licenses", []),
        "images": new_images,
        "annotations": new_anns,
        "categories": data.get("categories", [])
    }
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f"Wrote {out_path}: images={len(new_images)} anns={len(new_anns)} / total_images={n_total}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-root", required=True,
                    help="真实数据标注目录，例如：.../data/real_drone_ann")
    ap.add_argument("--out-root", default="",
                    help="输出目录（默认与 ann-root 相同）")
    ap.add_argument("--ratio", type=float, default=0.1,
                    help="抽取比例，默认 0.1")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，默认 42")
    ap.add_argument("--modalities", nargs="+",
                    choices=["visible", "infrared"], default=["visible", "infrared"],
                    help="选择要处理的模态，默认同时处理 visible 和 infrared")
    args = ap.parse_args()

    ann_root = osp.abspath(args.ann_root)
    out_root = osp.abspath(args.out_root) if args.out_root else ann_root
    pct_tag = f"{int(round(args.ratio*100))}pct"

    for mod in args.modalities:
        for split in ["val", "test"]:
            in_json = osp.join(ann_root, f"{split}_{mod}_full.json")
            if not osp.isfile(in_json):
                print(f"[WARN] 未找到 {in_json}，跳过。")
                continue
            out_json = osp.join(out_root, f"{split}_{mod}_{pct_tag}.json")
            subset_coco(in_json, out_json, args.ratio, args.seed)

if __name__ == "__main__":
    main()
