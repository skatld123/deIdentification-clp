# config.py

# Input Image 
root_testDir = '/root/dataset_clp/dataset_2044_new/test_only_one/'
ensemble_save_txt = '/root/deIdentification-clp/weighted_box_fusion/result/'
ensemble_save_img = "/root/deIdentification-clp/result/ensemble/"
crop_output_dir = '/root/deIdentification-clp/clp_landmark_detection/data/dataset/test/'

# Specify the path to model config and checkpoint file
config_file_1 = '/root/deIdentification-clp/weights/dino_2044_new_50/dino-5scale_swin-l_8xb2-36e_coco.py'
config_file_2 = 'yolo'
config_list = [config_file_1, config_file_2]

checkpoint_file_1 = '/root/deIdentification-clp/weights/dino_2044_new_50/best_coco_bbox_mAP_epoch_38.pth'
checkpoint_file_2 = '/root/deIdentification-clp/weights/yolov8/best_1280.pt'
checkpoint_file_list = [checkpoint_file_1, checkpoint_file_2]

cfg_ensemble = {
    'configs': config_list,
    'checkpoints': checkpoint_file_list,
    'input_img' : root_testDir + "images/",
    'input_lbl' : root_testDir + "labels/",
    'save_txt_dir': ensemble_save_txt + "predict/",
    'save_json_dir': ensemble_save_txt,
    'save_img': False,
    "save_img_dir" : ensemble_save_img,
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

