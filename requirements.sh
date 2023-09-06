# mmdet install
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e -r license_plate_mmdetection/requirements.txt

# ensemble install 
pip install ensemble-boxes
pip install natsort