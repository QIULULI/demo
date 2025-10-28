from mmengine.config import Config
from mmengine.registry import init_default_scope, TRANSFORMS
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS, DATASETS
from mmdet.structures.bbox import HorizontalBoxes

import torch, os, sys
from PIL import Image, ImageDraw
import numpy as np

# ===== 基本参数 =====
cfg_file = 'DG/Ours/drone/diffusion_detector_drone_rgb.py'
ckpt     = 'work_dirs/diffusion_detector_drone_rgb/best_coco_bbox_mAP_50_iter_3000.pth'
target_file_name = '00002/clear_day/matrix-300-RTK=m210-rtk/images_rgb/10.3753.png'  # val.json里的file_name

def _norm(p): 
    return os.path.normpath(str(p)).replace('\\', '/')

def extract_img_path(info):
    for k in ('img_path', 'file_name', 'filename'):
        if k in info and info[k]:
            return _norm(info[k])
    if 'img' in info and isinstance(info['img'], dict) and info['img'].get('img_path'):
        return _norm(info['img']['img_path'])
    if 'img_info' in info and isinstance(info['img_info'], dict):
        for k in ('filename', 'file_name'):
            if k in info['img_info'] and info['img_info'][k]:
                return _norm(info['img_info'][k])
    return None

def find_index_in_dataset(ds, target_rel):
    prefix = _norm(ds.data_prefix.get('img', '') or '')
    target_rel = _norm(target_rel).lstrip('/')
    for i in range(len(ds)):
        path = extract_img_path(ds.get_data_info(i))
        if not path: 
            continue
        path = _norm(path)
        # endswith/去前缀相对路径/同名文件 3重匹配
        if path.endswith('/' + target_rel): 
            return i
        if prefix and path.startswith(prefix):
            rel = _norm(path[len(prefix):]).lstrip('/')
            if rel == target_rel or rel.endswith('/' + target_rel):
                return i
        if os.path.basename(path) == os.path.basename(target_rel):
            return i
    return None

def boxes_to_numpy(boxes):
    if hasattr(boxes, 'tensor'):
        t = boxes.tensor
    elif isinstance(boxes, torch.Tensor):
        t = boxes
    else:
        t = torch.as_tensor(boxes)
    return t.detach().cpu().numpy()

def tensor_to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

# ===== 读取配置 / 构建VAL数据集（使用 test_pipeline）=====
cfg = Config.fromfile(cfg_file)
init_default_scope('mmdet')

test_pipeline = cfg.get('test_pipeline',
                        cfg.get('val_dataloader', {}).get('dataset', {}).get('pipeline', []))
assert test_pipeline, '配置中未找到 test_pipeline，请检查 cfg_file。'

val_ds_cfg = cfg.val_dataloader['dataset'].copy()
val_ds_cfg['pipeline']  = test_pipeline        # 确保无随机增广
val_ds_cfg['test_mode'] = True
val_ds = DATASETS.build(val_ds_cfg)

# 找目标样本
idx = find_index_in_dataset(val_ds, target_file_name)
if idx is None:
    print('在 val 数据集中未匹配到该 file_name。下面打印前 5 条样例的路径帮助排查：', file=sys.stderr)
    for j in range(min(5, len(val_ds))):
        print('example', j, '->', extract_img_path(val_ds.get_data_info(j)), file=sys.stderr)
    raise SystemExit(f'[FATAL] 找不到: {target_file_name}')

# ===== 取同一条样本（pipeline后：模型看到的图 + 同步后的GT）=====
sample = val_ds.prepare_data(idx)            # dict: 'inputs' & 'data_samples'
inputs = sample['inputs']                    # Tensor C,H,W
data_sample = sample['data_samples']         # DetDataSample
gt_boxes_np = boxes_to_numpy(data_sample.gt_instances.bboxes)

# 底图（pipeline后的图 = img_shape）
img_np = inputs.permute(1, 2, 0).numpy().astype('uint8')

# 关键元信息
meta = data_sample.metainfo
sf = np.array(meta.get('scale_factor', [1., 1., 1., 1.]), dtype=np.float32)
if sf.size == 2:  # 防守式
    sf = np.array([sf[0], sf[1], sf[0], sf[1]], dtype=np.float32)
ori_shape  = tuple(meta.get('ori_shape', ()))
img_shape  = tuple(meta.get('img_shape', ()))
batch_shape= tuple(meta.get('batch_input_shape', ()))

# ===== 构建模型并载入 ckpt（把 RPN 锚设回 [8]，与 ckpt 对齐）=====
cfg.model.setdefault('rpn_head', {}).setdefault('anchor_generator', {})['scales'] = [8]
model = MODELS.build(cfg.model)
_ = load_checkpoint(model, ckpt, map_location='cpu', revise_keys=[(r'^module\.', '')], strict=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()

# ===== 前向推理（dict；data_samples是list）=====
batch = {
    'inputs': inputs.unsqueeze(0).to(device),      # (1,C,H,W)
    'data_samples': [data_sample.to(device)]       # list[DetDataSample]
}
with torch.no_grad():
    out = model.test_step(batch)                   # list[DetDataSample]
pred = out[0]

# 预测框（可能已被rescale到ori_shape；我们同时给出两种画法）
pred_boxes_np  = boxes_to_numpy(pred.pred_instances.bboxes)
pred_scores_np = tensor_to_numpy(pred.pred_instances.scores)

# === 版本A：直接按返回的坐标画（as-is）
canvasA = Image.fromarray(img_np.copy()); drawA = ImageDraw.Draw(canvasA)
for x1, y1, x2, y2 in gt_boxes_np:
    drawA.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=(255, 0, 0), width=2)
for (x1, y1, x2, y2), s in zip(pred_boxes_np, pred_scores_np):
    if s < 0.3: continue
    drawA.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=(0, 255, 0), width=2)
canvasA.save('gt_pred_same_space_as_is.png')

# === 版本B：把预测从 ori_shape 映射回 img_shape 再画
# 假设 pred 返回在 ori_shape，则在 img_shape 上可用 boxes / scale_factor
sf4 = sf if sf.size == 4 else np.array([sf[0], sf[1], sf[0], sf[1]], dtype=np.float32)
pred_on_img = pred_boxes_np / (sf4 + 1e-6)

canvasB = Image.fromarray(img_np.copy()); drawB = ImageDraw.Draw(canvasB)
for x1, y1, x2, y2 in gt_boxes_np:
    drawB.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=(255, 0, 0), width=2)
for (x1, y1, x2, y2), s in zip(pred_on_img, pred_scores_np):
    if s < 0.3: continue
    drawB.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=(0, 255, 0), width=2)
canvasB.save('gt_pred_mapped_by_scale_factor.png')

print('saved -> gt_pred_same_space_as_is.png & gt_pred_mapped_by_scale_factor.png')
print(f'ori_shape={ori_shape}, img_shape={img_shape}, batch_input_shape={batch_shape}, scale_factor={sf.tolist()}')
