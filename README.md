# De-Identification for Car License Plate
## System Architecture
- Licnense Plate Detection Architecture using WBF
![검출구조](https://github.com/skatld123/deIdentification-clp/assets/79092711/e6a42ecd-d819-4340-ab39-e0b479d07ed9)
- De-identification Architecture
![비식별화 과정_수정](https://github.com/skatld123/deIdentification-clp/assets/79092711/311ff446-a989-4d64-8fb1-20882cea7ed5)

## Sub Module
- mmdetection(3.1.0)
- YOLOv8
- RetinaNet(for using CLP Landmark detection)
- De-Id(CycleGAN)

## Prepare
- Put your weights file in the weights path.
```
sh /root/deIdentification-clp/tools/download_dataset_and_weight.sh
```
- Prepare the dataset in the following format. (txt->YOLO, json->COCO)
- When testing only, the label is not essential.
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
3. download pretrained weight for lp detection, landmark_detection
```
sh download_dataset_and_weight.sh
```

## Install Directly
1. clone this project
```
git clone https://github.com/skatld123/deIdentification-clp.git
```
2. install denpendencies
```
cd deIdentification-clp
sh requirements.sh
```
3. You may need to install the cycleGAN dependency, if that's the case, please install the dependency recommended by [CycleGAN-Pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


## Using
1. Edit the config file to suit your dataset and path.
    - config/config_info.py
    - If the weight is yolo, you must enter "yolo" when adding it to config_list in the config_info.py_.
2. Run
```Shell
cd deIdentification-clp
python test.py
```

### Reference
- Detection : [mmdetection](https://github.com/open-mmlab/mmdetection), [yolov8](https://github.com/ultralytics/ultralytics), [WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [Virtual License Plate Image](https://github.com/Oh-JongJin/Virtual_Number_Plate)
- [CycleGAN-Pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
