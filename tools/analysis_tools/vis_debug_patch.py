# -*- coding: utf-8 -*-
# 目的：仅打印关键信息（包含图片路径），不改变可视化的绘制行为
# - 打印：img_path、image.shape、ori_shape、img_shape、pad_shape、scale_factor、前两条 GT（xyxy）
# - 兼容 HorizontalBoxes：在打印/绘制前转 .tensor
# - 不做任何 resize/pad 对齐处理（保持“原样可视化”），方便你复现和核对

import numpy as np

try:
    import torch
except Exception:
    torch = None

from mmdet.visualization.local_visualizer import DetLocalVisualizer

_orig_add = DetLocalVisualizer.add_datasample

def _to_numpy_boxes(b):
    if hasattr(b, 'tensor'):  # HorizontalBoxes/BaseBoxes
        b = b.tensor
    if torch is not None and isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return np.asarray(b)

def _add_datasample_probe(self, name, image, data_sample=None, **kwargs):
    meta = {}
    img_path = None
    ori_shape = None
    img_shape = None
    pad_shape = None
    scale_factor = None

    if data_sample is not None:
        try:
            meta = data_sample.metainfo or {}
        except Exception:
            meta = getattr(data_sample, 'metainfo', {}) or {}
        img_path    = meta.get('img_path', None)
        ori_shape   = meta.get('ori_shape', None)
        img_shape   = meta.get('img_shape', None)
        pad_shape   = meta.get('pad_shape', None)
        scale_factor= meta.get('scale_factor', None)

    # 读取前两条 GT（打印用）；绘制时也转成 tensor 以避免 HorizontalBoxes 兼容问题
    ds2 = data_sample
    gt_head = None
    try:
        if data_sample is not None and hasattr(data_sample, 'gt_instances'):
            b = data_sample.gt_instances.bboxes
            b_np = _to_numpy_boxes(b)
            if b_np.size:
                gt_head = b_np[:2]
            # 绘制时也用 tensor，避免可视化内部不兼容
            if hasattr(b, 'tensor'):
                ds2 = data_sample.clone()
                ds2.gt_instances.bboxes = b.tensor
    except Exception as e:
        print(f"[VIS-PROBE] (warn) read/convert gt failed: {e}")

    # 打印关键信息
    ishape = getattr(image, 'shape', None)
    print("[VIS-PROBE] name=%s" % (name,))
    print(f"[VIS-PROBE] path={img_path}")
    print(f"[VIS-PROBE] image.shape={ishape}")
    print(f"[VIS-PROBE] meta ori_shape={ori_shape} img_shape={img_shape} pad_shape={pad_shape} scale_factor={scale_factor}")
    if gt_head is not None:
        # 控制精度，避免太长
        with np.printoptions(precision=2, suppress=True):
            print(f"[VIS-PROBE] gt[:2]={gt_head}")
    else:
        print("[VIS-PROBE] gt[:2]=<empty>")

    # 调回原实现（不做任何图像对齐变换）
    return _orig_add(self, name, image, data_sample=ds2, **kwargs)

# 打补丁
DetLocalVisualizer.add_datasample = _add_datasample_probe
print("[VIS-PROBE] DetLocalVisualizer.add_datasample patched (print-only).")
# End of vis_debug_patch.py