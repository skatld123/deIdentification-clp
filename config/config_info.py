# config.py
# Input Image 
import os


# root_testDir = '/root/dataset_clp/dataset_v2/test'
root_testDir = '/root/deIdentification-clp/dataset/test_only_one'
gt_json_path = os.path.join(root_testDir, "test.json")

# Specify the path to model config and checkpoint file
one_stage_output = '/root/deIdentification-clp/result/one_stage_result/'
two_stage_output = '/root/deIdentification-clp/result/two_stage_result/'
ensemble_output = '/root/deIdentification-clp/result/ensemble_result/'
root_crop = "/root/deIdentification-clp/result/cropped_img"

# YOLOv8
cfg_one = {
    # AUG_best
    'configs' : 'yolo',
    'checkpoints' : '/root/deIdentification-clp/weights/1_stage/yolov8/best_1280_v2_aug_0.935.pt',
    'input_img' : os.path.join(root_testDir, "images/"),
    'input_lbl' : os.path.join(root_testDir, "labels/"),
    'output_img': os.path.join(one_stage_output, 'images/'),
    'output_lbl': os.path.join(one_stage_output, 'labels/'),
    'output_json' : os.path.join(one_stage_output, 'result.json'),
    'save_img' : False,
    'num2class' : {"0.0" : "license-plate", "1.0" : "vehicle"}
}


cfg_crop = {
    'input' : os.path.join(root_testDir,"images/"),
    'output_dir' : os.path.join(root_crop, 'vehicle'),
    'output' : os.path.join(root_crop, 'vehicle','images')
}

cfg_crop_lp = {
    'input' : os.path.join(root_testDir, "images/"),
    'output' : os.path.join(root_crop, 'license-plate','images')
}

# Swin_aug
cfg_two = {
    'configs' : '/root/deIdentification-clp/weights/2_stage/swin_crop_aug/swin_crop.py',
    'checkpoints': '/root/deIdentification-clp/weights/2_stage/swin_crop_aug/best_coco_bbox_mAP_epoch_5.pth',
    'input_img' : os.path.join(cfg_crop['output_dir'], 'images/'),
    'input_lbl' : os.path.join(cfg_crop['output_dir'], 'labels/'),
    'input_json' : os.path.join(cfg_crop['output_dir'], 'result.json'),
    'output_img': os.path.join(two_stage_output, 'images/'),
    'output_lbl' : os.path.join(two_stage_output, 'labels/'),
    'output_json' : os.path.join(two_stage_output ,'result.json'),
    'save_img' : True,
    'num2class' : {"0.0" : "license-plate"}
}

cfg_ensemble = {
    'input_img' : os.path.join(root_testDir, "images/"),
    'input_lbl' : os.path.join(root_testDir, "labels/"),
    'input_json' : os.path.join(root_testDir, "test.json"),
    'output_lbl': os.path.join(ensemble_output, 'labels/'),
    "output_img" : os.path.join(ensemble_output, 'images/'),
    'output_json': os.path.join(ensemble_output, 'result.json'),
    'save_img': True,
    "iou_thr" : 0.5,
    "skip_box_thr" : 0.0001,
    "sigma" : 0.1,
    'weight': [0.5, 1],
    'num2class' : {"0.0" : "license-plate", "1.0" : "vehicle"}
}

# LandmarkDetection
landmark_output_path = '/root/deIdentification-clp/clp_landmark_detection/results/'
lm_checkpoint_model = '/root/deIdentification-clp/weights/retinanet/Resnet50_Final.pth'
landmark_backbone = 'resnet50'  # or mobile0.25

cfg_landmark = {
    'backbone': 'resnet50', # or mobile0.25
    'checkpoint' : '/root/deIdentification-clp/weights/retinanet/Resnet50_Final.pth',
    'input_dir': cfg_crop_lp['output'],
    'output_dir': '/root/deIdentification-clp/result/landmark_result/',
    'checkpoint': lm_checkpoint_model,
    'nms_thr': 0.3, 
    'vis_thres' : 0.5,
    'imgsz' : 320
}

cfg_cyclegan = {
    'input_dir': cfg_crop_lp['output'],
    'output_dir': '/root/deIdentification-clp/result/de_id_result',
    'checkpoint' : '/root/deIdentification-clp/weights/cyclegan/license-plate_cyclegan_lsgan_v1'
}
