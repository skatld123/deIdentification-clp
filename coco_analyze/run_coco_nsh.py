import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoanalyze import COCOanalyze
import numpy as np
import skimage.io as io
import pylab

# refer by 
# https://github.com/matteorr/coco-analyze/blob/release/COCOanalyze_demo.ipynb
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

## set paths
dataDir  = '.'
dataType = 'val2014'
annType  = 'person_keypoints'
teamName = 'fakekeypoints100'


pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
# dataDir='../'
dataType='val2014'
annFile = '/root/dataset/dataset_4p_aug/%s_%s.json'%(prefix, dataType)
cocoGt=COCO(annFile)

#initialize COCO detections api
# resFile='/root/dataset/dataset_4p_aug/%s_%s_fake%s100_results.json'
# resFile='/root/dataset/dataset_4p_aug/predicts/%s_%s_results.json'
# resFile = resFile % (prefix, dataType)
result_path = '/root/dataset/dataset_4p_aug/predicts/json'
result_list = os.listdir(result_path)
for resFile in result_list :
    resFile = os.path.join(result_path, resFile)
    resFile = '/root/dataset/dataset_4p_aug/predicts/json/Resnet50_Final_keypoints_val2014_results_detection.json'
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    # imgIds=imgIds[0:100]
    print(len(imgIds))
    imgId = imgIds[np.random.randint(len(imgIds))]

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    coco_analyze = COCOanalyze(cocoGt, cocoDt, 'keypoints')
    os.makedirs(os.path.splitext(resFile)[0], exist_ok=True)
    coco_analyze.evaluate(verbose=True, makeplots=True, savedir=os.path.splitext(resFile)[0])

    # # analaze 
    # ## NOTE: the values below are all default

    # # set OKS threshold of the extended error analysis
    # coco_analyze.params.oksThrs       = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]

    # # set OKS threshold required to match a detection to a ground truth
    # coco_analyze.params.oksLocThrs    = .1

    # # set KS threshold limits defining jitter errors
    # coco_analyze.params.jitterKsThrs = [.5,.85]

    # # set the localization errors to analyze and in what order
    # # note: different order will show different progressive improvement
    # # to study impact of single error type, study in isolation
    # coco_analyze.params.err_types = ['miss','swap','inversion','jitter']

    # # area ranges for evaluation
    # # 'all' range is union of medium and large
    # coco_analyze.params.areaRng       = [[32 ** 2, 1e5 ** 2]] #[96 ** 2, 1e5 ** 2],[32 ** 2, 96 ** 2]
    # coco_analyze.params.areaRngLbl    = ['all'] # 'large','medium' 

    # coco_analyze.params.maxDets = [20]

    # coco_analyze.analyze(check_kpts=True, check_scores=True, check_bckgd=True)
    # coco_analyze.summarize(makeplots=True, savedir=os.path.splitext(resFile)[0])