# De-Identification for Car License Plate
## Sub Module
- mmdetection(3.1.0)
- YOLOv8
- RetinaNet(for using CLP Landmark detection)
- De-Id(openCV)

## Prepare
- Put your weights file in the weights path
- Prepare the dataset in the following format (txt->YOLO, json->COCO)
```
path/to/dataset
 - train
 - valid
 - test
   - images
     - 00..1.jpg
     - 00..1.jpg
     - 00..1.jpg
   - labels
     - 00..1.txt
     - 00..2.txt
     - 00..3.txt
   - test.json
```

## Using
1. Edit the config file to suit your dataset and path.
    - config/config_info.py
    - If the weight is yolo, you must enter "yolo" when adding it to config_list in the config_info.py_.
2. Run
```Shell
cd deIdentification-clp
python main.py --input path/to/dataset --output path/to/saveDir --gpu 0
```
