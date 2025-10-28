# -*- coding: utf-8 -*-
"""
按配置里的 VAL pipeline 仅画 GT，并打印详尽调试信息
- 兼容 MMDet 3.x
- 自动将 HorizontalBoxes 转为 Tensor/Numpy
- 如果官方可视化失败，自动用 OpenCV 兜底绘制，保证一定会生成图片
"""
import os, os.path as osp, argparse, pprint, numpy as np
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS, VISUALIZERS

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='配置文件路径，如 test_drone_coco.py')
    ap.add_argument('--output-dir', default='vis_gt_val_from_cfg', help='输出图片目录（会自动创建）')
    ap.add_argument('--max-num', type=int, default=50, help='最多处理多少张')
    ap.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置项（可选）')
    return ap.parse_args()

def to_numpy_img(x):
    # 支持 torch.Tensor(C,H,W) / np.ndarray(H,W,C)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                x = np.transpose(x, (1, 2, 0))
    except Exception:
        pass
    # 统一到 uint8 可视化
    if x.dtype != np.uint8:
        if x.max() <= 1.2:
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = x.clip(0, 255).astype(np.uint8)
    return x

# def boxes_to_numpy_xyxy(b):
#     """把各种形态的 bboxes 转成 numpy 的 (N,4) xyxy"""
#     try:
#         import torch
#         from mmengine.structures.bbox import BaseBoxes
#         if isinstance(b, BaseBoxes):
#             b = b.tensor
#         if isinstance(b, torch.Tensor):
#             b = b.detach().cpu().numpy()
#     except Exception:
#         # 没有 mmengine.structures 或 torch 的情况
#         pass
#     b = np.asarray(b)
#     return b
def boxes_to_numpy_xyxy(b):
    """把各种形态的 bboxes 转成 numpy 的 (N,4) xyxy"""
    # 关键：优先取 .tensor（兼容 HorizontalBoxes/BaseBoxes）
    if hasattr(b, 'tensor'):
        b = b.tensor
    try:
        import torch
        if isinstance(b, torch.Tensor):
            return b.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(b)

def draw_with_cv(img, bboxes, labels=None, out_file=None):
    """用 OpenCV 兜底绘制（绿色框）"""
    try:
        import cv2
    except Exception:
        cv2 = None
    im = img.copy()
    if bboxes is None or len(bboxes) == 0:
        if out_file is not None:
            os.makedirs(osp.dirname(out_file), exist_ok=True)
            try:
                from imageio import imwrite
                imwrite(out_file, im)
            except Exception:
                from PIL import Image
                Image.fromarray(im[:, :, ::-1]).save(out_file)
        return
    b = bboxes.astype(float)
    if cv2 is not None:
        for i, (x1, y1, x2, y2) in enumerate(b):
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            if labels is not None:
                cv2.putText(im, str(int(labels[i])), (int(x1), int(max(0, y1-5))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        cv2.imwrite(out_file, im)
    else:
        # 没有 cv2 就用 PIL 简单画
        from PIL import Image, ImageDraw
        pim = Image.fromarray(im[:, :, ::-1])
        dr = ImageDraw.Draw(pim)
        for (x1,y1,x2,y2) in b:
            dr.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        pim.save(out_file)

def main():
    args = parse_args()
    print('='*90)
    print('[INFO] Loading config:', args.config)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # ---------- 打印 VAL 数据集与 pipeline 关键字段 ----------
    ds_cfg = cfg.val_dataloader.dataset
    print('[INFO] Dataset cfg keys:', list(ds_cfg.keys()))
    print('[INFO] ann_file    :', ds_cfg.get('ann_file'))
    print('[INFO] data_root   :', ds_cfg.get('data_root', ''))
    print('[INFO] img_prefix  :', ds_cfg.get('data_prefix', {}).get('img', ''))
    pipeline = ds_cfg.get('pipeline', [])
    print('[INFO] VAL pipeline (in order):')
    for i, step in enumerate(pipeline):
        print(f'  [{i:02d}]', step.get('type'))
    idx_resize = next((i for i, s in enumerate(pipeline) if s.get('type') == 'Resize'), None)
    idx_loadann = next((i for i, s in enumerate(pipeline) if s.get('type') == 'LoadAnnotations'), None)
    if idx_resize is not None and idx_loadann is not None and idx_resize < idx_loadann:
        print('[WARN] ⚠️ 发现 Resize 在 LoadAnnotations 之前，这会导致可视化时 GT 偏移！请把 LoadAnnotations 放到 Resize 前面。')
    print('='*90)

    # ---------- 构建数据集 ----------
    dataset = DATASETS.build(ds_cfg)
    print('[INFO] Dataset class :', dataset.__class__.__name__)
    print('[INFO] Num samples   :', len(dataset))
    print('[INFO] Metainfo      :')
    try:
        pprint.pprint(getattr(dataset, 'metainfo', {}))
    except Exception:
        print('  (metainfo print failed)')

    # ---------- 构建可视化器 ----------
    vis = VISUALIZERS.build(dict(type='DetLocalVisualizer', name='vis'))
    try:
        vis.dataset_meta = getattr(dataset, 'metainfo', {})
    except Exception:
        pass

    os.makedirs(args.output_dir, exist_ok=True)
    print('[INFO] Output dir    :', osp.abspath(args.output_dir))
    print('='*90)

    n = min(len(dataset), args.max_num)
    ok, fail = 0, 0
    for i in range(n):
        try:
            data = dataset[i]
        except Exception as e:
            print(f'[ERR ] dataset[{i}] 取样失败：{e}')
            fail += 1
            continue

        # inputs / data_samples（新版）
        img = data.get('inputs', None)
        data_sample = data.get('data_samples', None)
        if img is None and 'img' in data:  # 兼容旧键
            img = data['img']
        img_np = to_numpy_img(img)

        # 拿 meta 和 gt
        meta = {}
        try:
            meta = getattr(data_sample, 'metainfo', {})
        except Exception:
            if isinstance(data_sample, dict):
                meta = data_sample.get('metainfo', {})

        img_path     = meta.get('img_path', None)
        ori_shape    = meta.get('ori_shape', None)
        img_shape    = meta.get('img_shape', None)
        pad_shape    = meta.get('pad_shape', None)
        scale_factor = meta.get('scale_factor', None)
        img_id       = meta.get('img_id', None)

        gt_inst = getattr(data_sample, 'gt_instances', None)
        if gt_inst is None and isinstance(data, dict):
            gt_inst = data.get('gt_instances', None)

        b_np = None
        labels_np = None
        num_gts = -1
        if gt_inst is not None:
            try:
                if hasattr(gt_inst, 'bboxes'):
                    b_np = boxes_to_numpy_xyxy(gt_inst.bboxes)
                if hasattr(gt_inst, 'labels'):
                    try:
                        import torch
                        labels = gt_inst.labels
                        if isinstance(labels, torch.Tensor):
                            labels_np = labels.detach().cpu().numpy()
                        else:
                            labels_np = np.asarray(labels)
                    except Exception:
                        labels_np = None
                num_gts = 0 if b_np is None else b_np.shape[0]
            except Exception as e:
                print(f'[WARN] 解析 gt_instances 失败：{e}')

        print(f'[SAMPLE {i:04d}] img_id={img_id} path={img_path}')
        print(f'  ori_shape={ori_shape} img_shape={img_shape} pad_shape={pad_shape} scale_factor={scale_factor}')
        print(f'  gt_count={num_gts}')
        if b_np is not None:
            print(f'  first_boxes (up to 3, xyxy): {np.array2string(b_np[:3], precision=2)}')

        out_file = osp.join(args.output_dir, f'{i:06d}.jpg')

        # 先尝试官方可视化（把 bboxes 转成 Tensor/ndarray 后塞回 data_sample 再画）
        vis_ok = False
        try:
            if b_np is not None:
                # clone & 替换为 numpy（或者你也可以替换成 .tensor）
                ds2 = data_sample.clone()
                # 不依赖类型检查，直接塞 numpy，DetLocalVisualizer 有分支能处理
                ds2.gt_instances.bboxes = b_np
            else:
                ds2 = data_sample
            vis.add_datasample(
                name=str(i),
                image=img_np,
                data_sample=ds2,
                draw_gt=True,
                draw_pred=False,
                show=False,
                out_file=out_file
            )
            vis_ok = True
            ok += 1
            print(f'  [OK ] saved -> {out_file} (visualizer)')
        except Exception as e:
            print(f'  [ERR] 可视化失败：{e}')
            fail += 1

        # 官方失败就兜底画一次，保证有图
        if not vis_ok:
            try:
                draw_with_cv(img_np[:, :, ::-1], b_np, labels_np, out_file)  # cv2 用 BGR，这里把 RGB->BGR
                print(f'  [OK ] saved -> {out_file} (fallback)')
                ok += 1
            except Exception as e:
                print(f'  [ERR] 兜底绘制也失败：{e}')

    print('='*90)
    print(f'[DONE] Tried {n} samples, success={ok}, fail={fail}.')
    print('      输出目录：', osp.abspath(args.output_dir))
    print('      请留意上面的 [WARN]/[ERR] 来定位问题。')

if __name__ == '__main__':
    main()
