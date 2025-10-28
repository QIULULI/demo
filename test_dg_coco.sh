#!/bin/bash
bash tools/dist_test_dg_coco.sh  \
DG/Ours/drone/diffusion_detector_drone_rgb.py \
work_dirs/diffusion_detector_drone_rgb/best_coco_bbox_mAP_50_iter_11000.pth \
1
# configs/diff/faster-rcnn_diff_fpn_1x_coco.py \
# work_models/faster-rcnn_diff_fpn_1x_coco.pth \
# 1

