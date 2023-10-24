# config.py
# Input Image 
root_testDir = '/root/dataset_clp/dataset_v2/test/'
ensemble_save_txt = '/root/deIdentification-clp/weighted_box_fusion/result/'
ensemble_save_img = "/root/deIdentification-clp/result/ensemble/"
crop_output_dir = '/root/deIdentification-clp/clp_landmark_detection/data/dataset/test/'

# Specify the path to model config and checkpoint file
one_stage_output = '/root/deIdentification-clp/result/one_stage_result/'
two_stage_output = '/root/deIdentification-clp/result/two_stage_result/'
ensemble_output = '/root/deIdentification-clp/result/ensemble_result/'

cfg_one = {
    'configs' : 'yolo',
    'checkpoints': '/root/deIdentification-clp/weights/yolov8/best_1280_v2.pt',
    'input_img' : root_testDir + "images/",
    'input_lbl' : root_testDir + "labels/",
    'output_img': one_stage_output + 'images/',
    'output_lbl': one_stage_output + 'labels/'
}

# cfg_one = {
#     'configs' : '/root/deIdentification-clp/weights/dino_v2/dino-5scale_swin-l_8xb2-36e_coco.py',
#     'checkpoints': '/root/deIdentification-clp/weights/dino_v2/best_coco_bbox_mAP_epoch_29.pth',
#     'input_img' : root_testDir + "images/",
#     'input_lbl' : root_testDir + "labels/",
#     'output_img': one_stage_output + 'images/',
#     'output_lbl': one_stage_output + 'labels/'
# }

cfg_crop = {
    'input' : root_testDir + "images/",
    'output' : '/root/deIdentification-clp/result/cropped_img/vehicle/images/'
}

cfg_crop_lp = {
    'input' : root_testDir + "images/",
    'output' : '/root/deIdentification-clp/result/cropped_img/license-plate/images/'
}

cfg_two = {
    'configs' : '/root/deIdentification-clp/weights/dino_crop/dino-5scale_swin-l_8xb2-36e_coco.py',
    'checkpoints': '/root/deIdentification-clp/weights/dino_crop/best_coco_bbox_mAP_epoch_27.pth',
    'input_img' : '/root/deIdentification-clp/result/cropped_img/vehicle/images/',
    'input_lbl' : '/root/deIdentification-clp/result/cropped_img/vehicle/labels/',
    'output_img': two_stage_output + 'images/',
    'output_lbl' : two_stage_output + 'labels/'
}

# cfg_two = {
#     'configs' : 'yolo',
#     'checkpoints': '/root/deIdentification-clp/weights/yolov8/best_640_crop.pt',
#     'input_img' : '/root/deIdentification-clp/result/cropped_img/vehicle/images/',
#     'input_lbl' : '/root/deIdentification-clp/result/cropped_img/vehicle/labels/',
#     'output_img': two_stage_output + 'images/',
#     'output_lbl' : two_stage_output + 'labels/',
#     'num2class' : {"0.0" : "license-plate"}
# }

cfg_ensemble = {
    'input_img' : root_testDir + "images/",
    'input_lbl' : root_testDir + "labels/",
    'output_lbl': ensemble_output + 'labels/',
    "output_img" : ensemble_output + 'images/',
    'output_json': ensemble_output,
    'save_img': False,
    "iou_thr" : 0.5,
    "skip_box_thr" : 0.0001,
    "sigma" : 0.1,
    'weight': [0.5, 1],
    'num2class' : {"0.0" : "license-plate", "1.0" : "vehicle"}
}

# LandmarkDetection
landmark_output_path = '/root/deIdentification-clp/clp_landmark_detection/results/'
lm_checkpoint_model = '/root/deIdentification-clp/clp_landmark_detection/weights/Resnet50_Final.pth'
landmark_backbone = 'resnet50'  # or mobile0.25

cfg_landmark = {
    'backbone': 'resnet50', # or mobile0.25
    'checkpoint' : '/root/deIdentification-clp/clp_landmark_detection/weights/Resnet50_Final.pth',
    'input_dir': crop_output_dir,
    'output_dir': landmark_output_path,
    'save_img' : True,
    'save_txt' : False,
    'checkpoint': lm_checkpoint_model,
    'nms_thr': 0.3, 
    'vis_thres' : 0.5,
    'imgsz' : 320
}

# yolo-yolo
# 0.862 -> 0.889 -> 0.918
# yolo-dino