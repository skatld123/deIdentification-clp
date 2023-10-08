from ultralytics import YOLO

# Load a model
# model = YOLO('/root/deIdentification-clp/yolo/license-plate/1280_scale_v22/weights/best.pt')  # build from YAML and transfer weights
model = YOLO('/root/license-plate/640_scale_crop/weights/best.pt')
# Validate the model
metrics = model.val(data='/root/dataset_clp/dataset_crop/data.yaml', imgsz=640, batch=16, device=[0,1], split='test',
                    save_txt=True, iou=0.5, conf=0.25)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category