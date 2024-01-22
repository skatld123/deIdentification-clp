from ultralytics import YOLO

# Load a model
model = YOLO('yolov8l.yaml').load('yolov8l.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/root/dataset_clp/dataset_v2_mask/data.yaml', epochs=150, imgsz=1280, device=[0,1], save=True, save_period=10,
                      batch=16, patience=20, workers=8, project="train_yolo", name="1280_v2_yolov8l", dropout=0.1)