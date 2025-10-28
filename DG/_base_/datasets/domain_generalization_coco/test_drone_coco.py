# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
classes = ('drone',)

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Pad', size_divisor=64),  # ← 新增：与模型里 pad_size_divisor 保持一致
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        #ann_file='drone_sim_coco/val.json',             # 改成你的 val/test.json
        ann_file='drone_ann/single_clear_day_ir/val.json',
        #data_prefix=dict(img='/userhome/liqiulu/data/drone_rgb_clear_day'),  # 与 COCO 中 file_name 对齐
        data_prefix=dict(img='/userhome/liqiulu/data/drone_ir_clear_day/00001'),  # ← 修改这里
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=test_pipeline,
        return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    #ann_file=data_root + 'drone_sim_coco/val.json',     # 同上修改
    ann_file=data_root + 'drone_ann/single_clear_day_ir/val.json',     # 同上修改
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
