_base_ = 'deformable-detr-refine_r50_16xb2-50e_coco.py'
model = dict(as_two_stage=True)

# learning policy
max_epochs = 100
work_dir = '/root/mmdetection/work_dirs/deformable_detr_twostage_' + str(max_epochs)

default_hooks = dict(
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=15,
        min_delta=0.005),
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        save_best='auto',
        out_dir=work_dir)
)

test_evaluator = dict(
    outfile_prefix='./work_dirs/clp_detection/deformable_detr_twostage_/')