#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, os.path as osp, argparse, json, pickle, glob, math
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm

def load_pkl(p): 
    with open(p, 'rb') as f: 
        return pickle.load(f)

def list_images(image_root, camera='rgb'):
    # 匹配 **/images_rgb/*.png
    pat = osp.join(image_root, '**', f'images_{camera}', '*.png')
    return sorted(glob.glob(pat, recursive=True))

def try_get_iminfo(drone_dir, camera='rgb'):
    # drone_dir 形如 .../DJI-avata2
    info_pkl = osp.join(drone_dir, 'im_info.pkl')
    if not osp.isfile(info_pkl):
        return None
    info = load_pkl(info_pkl)
    if camera not in info: 
        return None
    K = np.array(info[camera]['intrinsic'], dtype=float).reshape(3,3)
    E = np.array(info[camera]['extrinsic'], dtype=float).reshape(4,4)
    return K, E

def looks_like_instances(obj):
    # 期望：list of instances；每个 instance 是 [label(str), p1..p8] 且每个 p 是(3,)
    if not isinstance(obj, list) or len(obj)==0:
        return False
    def ok_inst(x):
        if not isinstance(x, (list,tuple)) or len(x) < 2: return False
        if not isinstance(x[0], str): return False
        for p in x[1:]:
            p = np.array(p)
            if p.shape != (3,): return False
        return True
    # 有的文件可能就是一个实例，外面再包一层
    return ok_inst(obj[0]) or (len(obj)==9 and isinstance(obj[0], str))

def normalize_instances(obj):
    # 统一成：list[ {label:str, pts3d:(N,3)} ]
    insts = []
    if isinstance(obj, list) and len(obj)==9 and isinstance(obj[0], str):
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
    z = x[:,2:3].copy()
    eps = 1e-6
    z[z==0] = eps
    uv = x[:,:2] / z
    return uv, x[:,2]  # (N,2), z

def to_homo(X):
    return np.concatenate([X, np.ones((X.shape[0],1))], axis=1)

def project_world(K, E, Xw, invE=False):
    """Xw: (N,3) 在世界坐标；E: 4x4; invE=True 时用 inv(E)"""
    T = np.linalg.inv(E) if invE else E
    Xw_h = to_homo(Xw)  # (N,4)
    Xc_h = (T @ Xw_h.T).T  # (N,4) 世界->相机（若E本身是相机->世界就 inv）
    Xc = Xc_h[:,:3]
    return project_cam(K, Xc)

def choose_projection(K, E, pts3d, img_w, img_h):
    """尝试三种假设：A相机系、B世界->相机(E)、C相机<-世界(invE)，选点有效数最多的一种"""
    modes = []
    # A: 已是相机坐标
    uvA, zA = project_cam(K, pts3d)
    validA = np.sum((zA>0) & (uvA[:,0]>=-50) & (uvA[:,1]>=-50) & (uvA[:,0]<img_w+50) & (uvA[:,1]<img_h+50))
    modes.append(('cam', uvA, zA, validA))
    # B: 世界->相机 直接用E
    uvB, zB = project_world(K, E, pts3d, invE=False)
    validB = np.sum((zB>0) & (uvB[:,0]>=-50) & (uvB[:,1]>=-50) & (uvB[:,0]<img_w+50) & (uvB[:,1]<img_h+50))
    modes.append(('w2c', uvB, zB, validB))
    # C: E是相机->世界，需要inv
    uvC, zC = project_world(K, E, pts3d, invE=True)
    validC = np.sum((zC>0) & (uvC[:,0]>=-50) & (uvC[:,1]>=-50) & (uvC[:,0]<img_w+50) & (uvC[:,1]<img_h+50))
    modes.append(('c2w(invE)', uvC, zC, validC))
    modes.sort(key=lambda x: x[3], reverse=True)
    return modes[0]  # best

def bbox_from_uv(uv, w, h, clip=True):
    x1, y1 = np.min(uv, axis=0)
    x2, y2 = np.max(uv, axis=0)
    if clip:
        x1 = max(0.0, min(float(x1), w-1))
        y1 = max(0.0, min(float(y1), h-1))
        x2 = max(0.0, min(float(x2), w-1))
        y2 = max(0.0, min(float(y2), h-1))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(bw), float(bh)]

def to_rel_path(path, base):
    import os.path as osp
    path = osp.abspath(path)
    base = osp.abspath(base)
    # 更稳健的相对路径获取（跨平台）
    rel = osp.relpath(path, base)
    return rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image_root', help='例如：/userhome/.../Town01_Opt/carla_data')
    ap.add_argument('-o','--out_dir', default='/userhome/liqiulu/data/drone_sim/ann', help='输出目录（保存 train.json / val.json）')
    ap.add_argument('--camera', default='rgb', choices=['rgb','ir','dvs'])
    ap.add_argument('--train-ratio', type=float, default=0.9)
    ap.add_argument('--min-box', type=float, default=4.0, help='最小 bbox 边长（像素）过滤')
    ap.add_argument('--max-images', type=int, default=0, help='仅调试看前N张(0为不限制)')
    ap.add_argument('--vis', type=int, default=0, help='可视化前N张到 out_dir/vis_debug')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = osp.join(args.out_dir, 'vis_debug')
    if args.vis>0: os.makedirs(vis_dir, exist_ok=True)

    img_list = list_images(args.image_root, args.camera)
    if args.max_images>0:
        img_list = img_list[:args.max_images]
    assert img_list, f'未在 {args.image_root} 下找到 images_{args.camera}/*.png'

    cat_name = 'drone'
    categories = [{'id': 1, 'name': cat_name}]
    images = []
    annos = []

    img_id = 0
    ann_id = 0

    # 缓存每个无人机目录的相机参数
    im_cache = {}

    for img_path in tqdm(img_list, desc='Converting'):
        # 对应 boxes 路径
        box_path = img_path.replace(f'images_{args.camera}', f'boxes_{args.camera}').replace('.png', '.pkl')
        if not osp.isfile(box_path):
            continue

        # 读取相机参数：.../DJI-xxx/images_rgb/xxx.png -> drone_dir = 上一级目录
        drone_dir = osp.dirname(osp.dirname(img_path))
        if drone_dir not in im_cache:
            iminfo = try_get_iminfo(drone_dir, args.camera)
            if iminfo is None:
                print(f'[WARN] 缺少 im_info.pkl 或 {args.camera} 相机参数: {drone_dir}')
                im_cache[drone_dir] = None
            else:
                im_cache[drone_dir] = iminfo
        if im_cache[drone_dir] is None:
            continue
        K, E = im_cache[drone_dir]

        # 图像尺寸
        im = cv2.imread(img_path)
        if im is None:
            print(f'[WARN] 读图失败: {img_path}')
            continue
        h, w = im.shape[:2]

        # 加载 boxes
        obj = load_pkl(box_path)
        if not looks_like_instances(obj):
            print(f'[WARN] 结构不符合预期: {box_path}')
            continue
        instances = normalize_instances(obj)

        # COCO images 项
        file_name = to_rel_path(img_path, args.image_root)
        images.append({
            'id': img_id,
            'file_name': file_name.replace('\\','/'),
            'height': h,
            'width':  w
        })

        # 每个目标 → 投影 → bbox
        for inst in instances:
            pts3d = inst['pts3d']  # (N,3)
            mode, uv, z, valid = choose_projection(K, E, pts3d, w, h)
            if valid < 4:  # 角点有效太少，放弃
                continue
            bbox = bbox_from_uv(uv, w, h, clip=True)
            if bbox[2] < args.min_box or bbox[3] < args.min_box:
                continue

            annos.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': [round(x,2) for x in bbox],
                'area': round(bbox[2]*bbox[3],2),
                'iscrowd': 0,
                'segmentation': []  # 检测任务可留空
            })
            ann_id += 1

        # 可视化少量样例
        if args.vis>0 and img_id < args.vis:
            vis = im.copy()
            import random
            for a in [x for x in annos if x['image_id']==img_id]:
                x,y,w2,h2 = map(int, a['bbox'])
                cv2.rectangle(vis, (x,y), (x+w2, y+h2), (0,255,0), 2)
            outp = osp.join(vis_dir, osp.basename(img_path))
            cv2.imwrite(outp, vis)

        img_id += 1

    # 划分 train/val
    n = len(images)
    idxs = list(range(n))
    # 固定顺序划分，保证可复现；也可以改成随机
    split = int(n * args.train_ratio)
    train_set = set(idxs[:split])
    val_set   = set(idxs[split:])

    def dump_json(which, keep_set):
        id_remap = {}
        imgs = []
        for i, img in enumerate(images):
            if i in keep_set:
                id_remap[img['id']] = len(imgs)
                img_cp = dict(img)
                img_cp['id'] = id_remap[img['id']]
                imgs.append(img_cp)
        anns = []
        for a in annos:
            if a['image_id'] in id_remap:
                b = dict(a)
                b['image_id'] = id_remap[a['image_id']]
                anns.append(b)
        js = {
            'info': {'description': 'DroneSim → COCO', 'version': '1.0'},
            'licenses': [],
            'images': imgs,
            'annotations': anns,
            'categories': categories
        }
        outp = osp.join(args.out_dir, f'{which}.json')
        with open(outp, 'w') as f:
            json.dump(js, f)
        print(f'Wrote {which}.json: images={len(imgs)} anns={len(anns)} -> {outp}')

    dump_json('train', train_set)
    dump_json('val',   val_set)

if __name__ == '__main__':
    main()
