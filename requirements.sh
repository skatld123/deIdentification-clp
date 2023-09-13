# mmdet install
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e -r license_plate_mmdetection/requirements.txt
mim install mmdet

# submodule load
git submodule init 
git submodule update
# git submodule foreach git checkout submodule

# register custom models
cd license_plate_mmdetection
pip install -v -e .

# ensemble install 
pip install ensemble-boxes
pip install natsort

# install yolo
pip install ultralytics