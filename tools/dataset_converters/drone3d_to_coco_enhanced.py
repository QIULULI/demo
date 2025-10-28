#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drone3D → COCO (enhanced)
- 支持相机选择：rgb/ir/dvs
- 支持筛选：--weather, --model
- 支持拆分：--split-by none|weather|model|both
- 支持软链接重整：--symlink-root [--symlink-boxes]
- 新增：支持全局相机参数 PKL：--iminfo-pkl
- 保持向后兼容：不加新参数时，行为等同于旧版（输出 train.json / val.json）
"""
import os, os.path as osp, argparse, json, pickle, glob
import numpy as np
import cv2
from tqdm import tqdm

# -------------------- 基础工具 --------------------

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

def list_images(image_root, camera='rgb'):
    """匹配 **/images_<camera>/*.png"""
    pat = osp.join(image_root, '**', f'images_{camera}', '*.png')
    return sorted(glob.glob(pat, recursive=True))

def try_get_iminfo(drone_dir, camera='rgb'):
    """（旧逻辑）从每个 <model_dir> 里读 im_info.pkl。"""
    info_pkl = osp.join(drone_dir, 'im_info.pkl')
    if not osp.isfile(info_pkl):
        return None
    info = load_pkl(info_pkl)
    if camera not in info:
        return None
    K = np.array(info[camera]['intrinsic'], dtype=float).reshape(3, 3)
    E = np.array(info[camera]['extrinsic'], dtype=float).reshape(4, 4)
    return K, E

def load_global_iminfo(pkl_path, camera='rgb'):
    """（新逻辑）从固定路径读取全局相机参数，所有图片共用。"""
    assert osp.isfile(pkl_path), f'全局 im_info.pkl 不存在：{pkl_path}'
    info = load_pkl(pkl_path)
    assert camera in info, f'全局 im_info.pkl 中不存在相机键：{camera}；可用键：{list(info.keys())}'
    K = np.array(info[camera]['intrinsic'], dtype=float).reshape(3, 3)
    E = np.array(info[camera]['extrinsic'], dtype=float).reshape(4, 4)
    return K, E

def looks_like_instances(obj):
    """期望：list of instances；每个 instance 是 [label(str), p1..p8] 且每个 p 是(3,)"""
    if not isinstance(obj, list) or len(obj) == 0:
        return False

    def ok_inst(x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            return False
        if not isinstance(x[0], str):
            return False
        for p in x[1:]:
            p = np.array(p)
            if p.shape != (3,):
                return False
        return True

    return ok_inst(obj[0]) or (len(obj) == 9 and isinstance(obj[0], str))

def normalize_instances(obj):
    """统一成：list[ {label:str, pts3d:(N,3)} ]"""
    insts = []
    if isinstance(obj, list) and len(obj) == 9 and isinstance(obj[0], str):
        obj = [obj]  # 单实例 → 列表
    for item in obj:
        label = item[0]
        pts = [np.array(p, dtype=float).reshape(3) for p in item[1:]]
        pts = np.stack(pts, axis=0)  # (N,3)
        insts.append({'label': label, 'pts3d': pts})
    return insts

def project_cam(K, Xc):
    """Xc: (N,3) 已在相机坐标系"""
    x = (K @ Xc.T).T  # (N,3)
    z = x[:, 2:3].copy()
    eps = 1e-6
    z[z == 0] = eps
    uv = x[:, :2] / z
    return uv, x[:, 2]  # (N,2), z

def to_homo(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

def project_world(K, E, Xw, invE=False):
    """Xw: (N,3) 在世界坐标；E: 4x4; invE=True 时用 inv(E)"""
    T = np.linalg.inv(E) if invE else E
    Xw_h = to_homo(Xw)  # (N,4)
    Xc_h = (T @ Xw_h.T).T  # (N,4) 世界->相机（若E本身是相机->世界就 inv）
    Xc = Xc_h[:, :3]
    return project_cam(K, Xc)

def choose_projection(K, E, pts3d, img_w, img_h):
    """尝试三种假设：A相机系、B世界->相机(E)、C相机<-世界(invE)，选点有效数最多的一种"""
    modes = []
    # A: 已是相机坐标
    uvA, zA = project_cam(K, pts3d)
    validA = np.sum((zA > 0) & (uvA[:, 0] >= -50) & (uvA[:, 1] >= -50) &
                    (uvA[:, 0] < img_w + 50) & (uvA[:, 1] < img_h + 50))
    modes.append(('cam', uvA, zA, validA))
    # B: 世界->相机 直接用E
    uvB, zB = project_world(K, E, pts3d, invE=False)
    validB = np.sum((zB > 0) & (uvB[:, 0] >= -50) & (uvB[:, 1] >= -50) &
                    (uvB[:, 0] < img_w + 50) & (uvB[:, 1] < img_h + 50))
    modes.append(('w2c', uvB, zB, validB))
    # C: E是相机->世界，需要inv
    uvC, zC = project_world(K, E, pts3d, invE=True)
    validC = np.sum((zC > 0) & (uvC[:, 0] >= -50) & (uvC[:, 1] >= -50) &
                    (uvC[:, 0] < img_w + 50) & (uvC[:, 1] < img_h + 50))
    modes.append(('c2w(invE)', uvC, zC, validC))
    modes.sort(key=lambda x: x[3], reverse=True)
    return modes[0]  # best

def bbox_from_uv(uv, w, h, clip=True):
    x1, y1 = np.min(uv, axis=0)
    x2, y2 = np.max(uv, axis=0)
    if clip:
        x1 = max(0.0, min(float(x1), w - 1))
        y1 = max(0.0, min(float(y1), h - 1))
        x2 = max(0.0, min(float(x2), w - 1))
        y2 = max(0.0, min(float(y2), h - 1))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(bw), float(bh)]

def to_rel_path(path, base):
    path = osp.abspath(path)
    base = osp.abspath(base)
    rel = osp.relpath(path, base)
    return rel

# -------------------- 新增：weather/model 解析 & 软链工具 --------------------

_KNOWN_WEATHERS = {
    'clear_day', 'clear_night',
    'fog_day', 'fog_night',
    'rain_day', 'rain_night',
    'snow_day', 'snow_night'
}

def parse_meta_from_path(img_path, image_root, camera):
    """Return (weather, model_dir, relpath_from_root).
    假设路径形如：.../<weather>/<model_dir>/images_<camera>/xxx.png
    不符合时回退到 'unknown'。
    """
    rel = to_rel_path(img_path, image_root)
    parts = rel.split(osp.sep)
    key = f'images_{camera}'
    if key in parts:
        idx = parts.index(key)
    else:
        # fallback: 尝试最后一个 images_* 位置
        idx = len(parts) - 2 if len(parts) >= 2 else 0
    model_dir = parts[idx - 1] if idx - 1 >= 0 else 'unknown_model'
    weather = parts[idx - 2] if idx - 2 >= 0 else 'unknown'
    if weather not in _KNOWN_WEATHERS:
        weather = 'unknown'
    return weather, model_dir, rel

def safe_symlink(src, dst):
    """创建软链 dst -> src。若失败尝试硬链，再不行则复制。"""
    import shutil
    os.makedirs(osp.dirname(dst), exist_ok=True)
    try:
        if osp.islink(dst):
            if osp.realpath(dst) == osp.realpath(src):
                return
            os.remove(dst)
        elif osp.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    except OSError:
        try:
            if osp.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

# -------------------- 主流程 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image_root', help='例如：/userhome/.../Town01_Opt/carla_data')
    ap.add_argument('-o', '--out_dir', default='/userhome/liqiulu/data/drone_sim/ann',
                    help='输出目录（保存 train*.json / val*.json）')
    ap.add_argument('--camera', default='rgb', choices=['rgb', 'ir', 'dvs'])
    ap.add_argument('--train-ratio', type=float, default=0.9)
    ap.add_argument('--min-box', type=float, default=4.0, help='最小 bbox 边长（像素）过滤')
    ap.add_argument('--max-images', type=int, default=0, help='仅调试看前N张(0为不限制)')
    ap.add_argument('--vis', type=int, default=0, help='可视化前N张到 out_dir/vis_debug')
    # 新增能力
    ap.add_argument('--weather', nargs='*', default=None,
                    help='仅保留这些天气(空表示全部)，如: clear_day fog_night')
    ap.add_argument('--model', nargs='*', default=None,
                    help='仅保留这些机型/组合(目录名)，如: matrix-300-RTK=drone-unk3 DJI-avata2')
    ap.add_argument('--split-by', choices=['none', 'weather', 'model', 'both'], default='none',
                    help='按天气/机型拆成多份 COCO')
    ap.add_argument('--symlink-root', default='',
                    help='若指定，将把选中的图片(及可选boxes)软链接到该目录，并让 COCO file_name 相对它写入')
    ap.add_argument('--symlink-boxes', action='store_true',
                    help='同时软链接 boxes_*.pkl（一般不需要）')
    # ===== 新增：全局相机参数 PKL =====
    ap.add_argument('--iminfo-pkl', default='',
                    help='全局 im_info.pkl 的绝对路径；指定后所有图片共用该相机参数（不再逐目录查找）')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = osp.join(args.out_dir, 'vis_debug')
    if args.vis > 0:
        os.makedirs(vis_dir, exist_ok=True)
    if args.symlink_root:
        os.makedirs(args.symlink_root, exist_ok=True)

    # 若提供了全局 im_info.pkl，则先加载
    global_iminfo = None
    if args.iminfo_pkl:
        try:
            global_iminfo = load_global_iminfo(args.iminfo_pkl, args.camera)
        except AssertionError as e:
            raise SystemExit(str(e))
        except Exception as e:
            raise SystemExit(f'读取全局 im_info.pkl 失败：{e}')

    # 收集图片
    img_list = list_images(args.image_root, args.camera)
    if args.max_images > 0:
        img_list = img_list[:args.max_images]
    assert img_list, f'未在 {args.image_root} 下找到 images_{args.camera}/*.png'

    # 分组 key 及容器
    def group_key(weather, model_dir):
        if args.split_by == 'weather':
            return (weather,)
        if args.split_by == 'model':
            return (model_dir,)
        if args.split_by == 'both':
            return (weather, model_dir)
        return ('all',)

    groups = {}  # key -> dict(images=[], anns=[], next_img_id=1, next_ann_id=1)
    def ensure_group(key):
        if key not in groups:
            groups[key] = dict(images=[], anns=[], next_img_id=1, next_ann_id=1)
        return groups[key]

    categories = [{'id': 1, 'name': 'drone'}]
    im_cache = {}  # drone_dir -> (K,E)（仅当未提供 --iminfo-pkl 时使用）

    kept = 0
    for img_path in tqdm(img_list, desc='converting'):
        weather, model_dir, rel_from_root = parse_meta_from_path(img_path, args.image_root, args.camera)

        # 过滤
        if args.weather and weather not in args.weather:
            continue
        if args.model and model_dir not in args.model:
            continue

        # boxes 路径
        box_path = img_path.replace(f'images_{args.camera}', f'boxes_{args.camera}').replace('.png', '.pkl')
        if not osp.isfile(box_path):
            continue

        # 相机参数
        if global_iminfo is not None:
            K, E = global_iminfo
        else:
            # 旧逻辑：上两级目录（.../<model_dir>/images_*/*.png）
            drone_dir = osp.dirname(osp.dirname(img_path))
            if drone_dir not in im_cache:
                iminfo = try_get_iminfo(drone_dir, args.camera)
                im_cache[drone_dir] = iminfo if iminfo is not None else None
                if iminfo is None:
                    print(f'[WARN] 缺少 im_info.pkl 或 {args.camera} 相机参数: {drone_dir}')
            if im_cache[drone_dir] is None:
                continue
            K, E = im_cache[drone_dir]

        # 读图 & 尺寸
        im = cv2.imread(img_path)
        if im is None:
            print(f'[WARN] 读图失败: {img_path}')
            continue
        h, w = im.shape[:2]

        # 读 boxes
        obj = load_pkl(box_path)
        if not looks_like_instances(obj):
            print(f'[WARN] 结构不符合预期: {box_path}')
            continue
        instances = normalize_instances(obj)

        # 选择分组
        key = group_key(weather, model_dir)
        G = ensure_group(key)

        # 若指定软链接：把图片(与可选 boxes)软链到新目录，并让 file_name 相对新目录
        if args.symlink_root:
            if args.split_by == 'none':
                tag = ''
            elif args.split_by == 'weather':
                tag = f'weather={weather}'
            elif args.split_by == 'model':
                tag = f'model={model_dir}'
            else:  # both
                tag = f'weather={weather}/model={model_dir}'
            dst_rel = osp.join(tag, rel_from_root) if tag else rel_from_root
            dst_abs = osp.join(args.symlink_root, dst_rel)
            safe_symlink(img_path, dst_abs)
            if args.symlink_boxes:
                dst_box_rel = dst_rel.replace(f'images_{args.camera}', f'boxes_{args.camera}').replace('.png', '.pkl')
                dst_box_abs = osp.join(args.symlink_root, dst_box_rel)
                safe_symlink(box_path, dst_box_abs)
            file_name = dst_rel.replace('\\', '/')
        else:
            file_name = rel_from_root.replace('\\', '/')

        # 写入 images
        img_id = G['next_img_id']; G['next_img_id'] += 1
        G['images'].append({
            'id': img_id,
            'file_name': file_name,
            'height': h,
            'width': w
        })

        # 逐实例 → 投影 → bbox
        for inst in instances:
            pts3d = inst['pts3d']
            mode, uv, z, valid = choose_projection(K, E, pts3d, w, h)
            if valid < 4:
                continue
            bbox = bbox_from_uv(uv, w, h, clip=True)
            if bbox[2] < args.min_box or bbox[3] < args.min_box:
                continue
            ann_id = G['next_ann_id']; G['next_ann_id'] += 1
            G['anns'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': [round(x, 2) for x in bbox],
                'area': round(bbox[2] * bbox[3], 2),
                'iscrowd': 0,
                'segmentation': []
            })

        kept += 1
        # 可视化少量样例
        if args.vis > 0 and kept <= args.vis:
            vis = im.copy()
            for a in [x for x in G['anns'] if x['image_id'] == img_id]:
                x, y, w2, h2 = map(int, a['bbox'])
                cv2.rectangle(vis, (x, y), (x + w2, y + h2), (0, 255, 0), 2)
            cv2.putText(vis, f'{weather} | {model_dir}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            vis_p = osp.join(vis_dir, file_name.replace('/', '__'))
            os.makedirs(osp.dirname(vis_p), exist_ok=True)
            cv2.imwrite(vis_p, vis)

    # 分组写 JSON（并做 train/val 切分）
    def write_group_json(key, G):
        # 稳定划分：按 file_name 排序
        imgs_sorted = sorted(G['images'], key=lambda d: d['file_name'])
        n_train = int(len(imgs_sorted) * args.train_ratio)
        imgset = {
            'train': set(d['id'] for d in imgs_sorted[:n_train]),
            'val':   set(d['id'] for d in imgs_sorted[n_train:])
        }
        for which in ['train', 'val']:
            keep = imgset[which]
            out_images = [d.copy() for d in imgs_sorted if d['id'] in keep]
            out_anns = [a.copy() for a in G['anns'] if a['image_id'] in keep]
            # 重新编号（更整洁）
            img_id_map = {d['id']: i + 1 for i, d in enumerate(out_images)}
            for i, d in enumerate(out_images):
                d['id'] = i + 1
            for i, a in enumerate(out_anns):
                a['id'] = i + 1
                a['image_id'] = img_id_map[a['image_id']]
            js = {
                'info': {'description': 'DroneSim → COCO', 'version': '1.1'},
                'licenses': [],
                'images': out_images,
                'annotations': out_anns,
                'categories': categories
            }
            if key == ('all',):
                tag = 'all'
            elif len(key) == 1:
                tag = f'weather-{key[0]}' if args.split_by == 'weather' else f'model-{key[0]}'
            else:
                tag = f'weather-{key[0]}__model-{key[1]}'
            outp = osp.join(args.out_dir,
                            f'{which}_{tag}.json' if args.split_by != 'none' else f'{which}.json')
            with open(outp, 'w') as f:
                json.dump(js, f)
            print(f'Wrote {osp.basename(outp)}: images={len(out_images)} anns={len(out_anns)} -> {outp}')

    for key, G in groups.items():
        write_group_json(key, G)

    if not groups:
        print('[WARN] 没有符合筛选条件的样本；请检查 --camera/--weather/--model 参数或数据目录。')

if __name__ == '__main__':
    main()
