from ultralytics import YOLO

# Load a model
# build from YAML and transfer weights
model = YOLO('/root/deIdentification-clp/weights/yolov8/best_1280.pt')
# Validate the model
metrics = model.val(data='/root/dataset_clp/dataset_2044_new/data.yaml', imgsz=1280, batch=16, device=[0, 1], split='test',
                    save_txt=True, iou=0.5, conf=0.25)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
