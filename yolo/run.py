from ultralytics import YOLO
# import clearml

# clearml.browser_login()
# Load a model
# model = YOLO('yolov8X.yaml')  # build a new model from YAML
# model = YOLO('yolov8X.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

# Train the model
# results = model.train(data='/root/dataset_clp/dataset_crop/data.yaml', epochs=200, imgsz=640, device=[0,1], save_period=10,
#                       batch=32, patience=20, workers=8, project="license-plate", name="640_scale_crop", dropout=0.1)
results = model.train(data='/root/dataset_clp/dataset_v2/data.yaml', epochs=100, imgsz=1280, device=[0,1], save=True, save_period=5,
                      batch=16, patience=20, workers=8, project="YOLO_train", name="1280_v2", dropout=0.1)