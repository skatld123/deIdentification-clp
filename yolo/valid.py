from ultralytics import YOLO

# Load a model

model = YOLO('/root/deIdentification-clp/weights/yolov8/best_1280_v2_aug_0.935.pt')
metrics = model.val(data='/root/dataset_clp/dataset_v2/data.yaml', imgsz=1280, batch=16, device=[0,1], split='test',
                    save_txt=False, iou=0.5, conf=0.25) 
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category