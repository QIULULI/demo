# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp

import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmengine.fileio import dump
from mmengine.utils import (Timer, mkdir_or_exist, track_parallel_progress,
                            track_progress)


def collect_files(img_dir, gt_dir, suffixes):
    """收集图片与对应 GT。suffixes 可为多个后缀。"""
    if isinstance(suffixes, str):
        suffixes = [s.strip() for s in suffixes.split(',') if s.strip()]

    files = []
    # ** 需要 recursive=True 才能递归
    for img_file in glob.glob(osp.join(img_dir, '**', '*.png'), recursive=True):
        if not any(img_file.endswith(suf) for suf in suffixes):
            continue
        # 用匹配到的后缀确定替换长度
        suf = next(s for s in suffixes if img_file.endswith(s))
        inst_file = gt_dir + img_file[len(img_dir):-len(suf)] + 'gtFine_instanceIds.png'
        segm_file = gt_dir + img_file[len(img_dir):-len(suf)] + 'gtFine_labelIds.png'
        files.append((img_file, inst_file, segm_file))
    assert len(files), f'No images found in {img_dir} with suffixes: {suffixes}'
    print(f'Loaded {len(files)} images from {img_dir}')
    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = track_parallel_progress(load_img_info, files, nproc=nproc)
    else:
        images = track_progress(load_img_info, files)
    return images


def load_img_info(files):
    img_file, inst_file, segm_file = files
    inst_img = mmcv.imread(inst_file, 'unchanged')
    unique_inst_ids = np.unique(inst_img[inst_img >= 24])
    anno_info = []
    for inst_id in unique_inst_ids:
        label_id = inst_id // 1000 if inst_id >= 1000 else inst_id
        label = CSLabels.id2label[label_id]
        if not label.hasInstances or label.ignoreInEval:
            continue
        category_id = label.id
        iscrowd = int(inst_id < 1000)
        mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]
        area = maskUtils.area(mask_rle)
        bbox = maskUtils.toBbox(mask_rle)
        mask_rle['counts'] = mask_rle['counts'].decode()
        anno_info.append(dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle))
    video_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        file_name=osp.join(video_name, osp.basename(img_file)),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(video_name, osp.basename(segm_file)))
    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict(info=dict(version='1.0', description='Cityscapes->COCO by script'))  # 补上 info，避免后续评测 KeyError
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in CSLabels.labels:
        if label.hasInstances and not label.ignoreInEval:
            out_json['categories'].append(dict(id=label.id, name=label.name))
    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')
    dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to COCO format')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--img-dir', default='leftImg8bit', type=str)
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('--img-suffix', default='leftImg8bit.png',
                        help='支持逗号分隔的多个后缀，例如：leftImg8bit.png,leftImg8bit_foggy_beta_0.005.png')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='只转换存在的 split；Foggy 通常是 train/val')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--nproc', default=1, type=int, help='number of process')
    return parser.parse_args()


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mkdir_or_exist(out_dir)

    img_dir = osp.join(cityscapes_path, args.img_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    set_name = {s: f'instancesonly_filtered_gtFine_{s}.json' for s in args.splits}

    for split, json_name in set_name.items():
        img_split = osp.join(img_dir, split)
        gt_split = osp.join(gt_dir, split)
        if not osp.isdir(img_split):
            print(f'[{split}] skipped: {img_split} not found.')
            continue
        print(f'Converting {split} into {json_name}')
        with Timer(print_tmpl='It took {}s to convert Cityscapes annotation'):
            files = collect_files(img_split, gt_split, args.img_suffix)
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
