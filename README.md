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
## Install
### Using docker 
1. pull docker image
```
docker pull skatld802/deid-clp:1.0
```
2. run docker container
```
docker run -d -it --gpus all --name de_id_clp --ipc=host -v {dataset_path}:{container_dataset_path} skatld802/de-id-clp:1.0
```


### Install Directly
1. clone this project
```
git clone https://github.com/skatld123/deIdentification-clp.git
```
2. install denpendencies
```
cd deIdentification-clp
sh requirements.sh
```


## Using
1. Edit the config file to suit your dataset and path.
    - config/config_info.py
    - If the weight is yolo, you must enter "yolo" when adding it to config_list in the config_info.py_.
2. Run
```Shell
cd deIdentification-clp
python main.py --input path/to/dataset --output path/to/saveDir
```
