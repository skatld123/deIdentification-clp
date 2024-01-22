import argparse
import ast
import csv
import json
import os

import clp_landmark_detection.detect as landmark
import config as cf
import cv2
import numpy as np
import weighted_box_fusion.clp_ensemble as ensemble
from deId import deIdentify
from make_detection_result_coco import custom_to_coco_result
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from weighted_box_fusion.utils import (boundingBoxes, boxPlot, cal_mAP,
                                       convert_scaled_predict)


# import coco_analyze
def parse_args():
    parser = argparse.ArgumentParser(description='DeIdentify Dataset')
    parser.add_argument(
        '--dataset', default="/root/dataset_clp/dataset_v2/test_only_one/", help='Test Dataset Path')
    parser.add_argument('--output', default="/root/deIdentification-clp/result/de_id_result", help='DeID output Directory')
    parser.add_argument('--gpu', default=0, help='Avaliable GPU Node')
    args = parser.parse_args()
    return args

def cropping_image(input_dir, output_dir, detections, cls, save_img) :
    dic_bbox_with_point = {}
    print("Cropping only LicensePlate Bounding Box from Input Image")
    index = 0
    if not os.path.exists(output_dir) : os.mkdir(output_dir)
    for idx, detection in enumerate(tqdm(detections)):
        # 박스 변환
        if isinstance(detection[3], tuple) : box = list(detection[3])
        else : box = ast.literal_eval(detection[3])

        file_name, label, conf = detection[0], float(detection[1]), float(detection[2])
        if label == cls:
            img_path = os.path.join(input_dir, file_name + ".jpg")
            img = Image.open(img_path)
            cropped_img = img.crop(box)
            if save_img :
                cropped_img.save(f'{output_dir}{file_name}_crop_{index:04d}.jpg')
            if file_name in dic_bbox_with_point:
                dic_bbox_with_point[file_name]['subimage'].append(
                    f'{file_name}_crop_{index:04d}.jpg')
                dic_bbox_with_point[file_name]['boxes'].append(box)
            else:
                dic_bbox_with_point[file_name] = {'subimage': [
                    f'{file_name}_crop_{index:04d}.jpg'], 'boxes': [box]}
            index += 1
        else:
            continue
    print(
        f"Cropped result images saved at : {output_dir} \nCropped result images cnt : {len(os.listdir(output_dir))}")
    return dic_bbox_with_point

def point_local2global(bbox_with_point, detection_result) :
    # detection result xyxy임
    global_dic = {}
    for key in bbox_with_point :
        img = cv2.imread(os.path.join(cf.cfg_crop['input'], key + ".jpg"))
        mh, mw, _ = img.shape
        subImgList = bbox_with_point[key]['subimage']
        subImgboxes = bbox_with_point[key]['boxes']
        for subImgName, subImgbox in zip(subImgList, subImgboxes) : 
            sx1, sy1, sx2, sy2 = subImgbox
            sw = sx2 - sx1
            sh = sy2 - sy1
            if os.path.splitext(subImgName)[0] in detection_result :
                cropImg = detection_result[os.path.splitext(subImgName)[0]]
            else :
                print(f"no detection in {os.path.splitext(subImgName)[0]}")
                continue
            boxes, scores, labels = cropImg['boxes'], cropImg['scores'], cropImg['labels']
            for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)) :
                rx1, ry1, rx2, ry2 = box
                x1, y1, x2, y2 = convert_scaled_predict((sw, sh), (rx1, ry1, rx2, ry2))
                mx1, my1, mx2, my2 = (x1 + sx1) / mw, (y1 + sy1) / mh, (x2 + sx1) / mw, (y2 + sy1) / mh
                if key in global_dic :
                    global_dic[key]['boxes'].append([mx1, my1, mx2, my2])
                    global_dic[key]['scores'].append(score)
                    global_dic[key]['labels'].append(label)
                else : 
                    global_dic[key] = {'boxes' : [[mx1, my1, mx2, my2]], 'scores' : [score], 'labels' : [label]}
                # cv2.rectangle(img, (mx1, my1), (mx2, my2), (255,0,0), 4)
        # cv2.imwrite(f"/root/deIdentification-clp/result/{key}.jpg", img)
    return global_dic

def coco_evaluate_offline(gt_path, pd_path, output_path) :
    if os.path.exists(gt_path) :
        custom_to_coco_result(gt_path, pd_path, output_path)
    else :
        print(f"ERROR : gt json file {gt_path} does not exist")
        return
    
    if not os.path.exists(output_path) : 
        print(f"{output_path} can not found")
        exit()
    else :
        cocoGt=COCO(gt_path)
        cocoDt=cocoGt.loadRes(output_path)
        imgIds=sorted(cocoGt.getImgIds())
        # imgIds=imgIds[0:100]
        print(f"Start COCO Evaluation.. Image Lenth : {len(imgIds)}")
        print(f"Ground Truth Json : {gt_path}")
        print(f"Detection Result Json : {output_path}")
        imgId = imgIds[np.random.randint(len(imgIds))]
        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
        # cocoEval.params.maxDets = [300]
        print("License-Plate")
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        print("Vehicle")
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds = [2]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

if __name__ == '__main__':
    args = parse_args()
    deid_output = args.output
    cf.root_testDir = args.dataset
    
    current_directory = os.getcwd()
    print("현재 작업 디렉토리:", current_directory)
    
    # De-ID Result
    cfg_en = cf.cfg_ensemble
    cfg_one = cf.cfg_one
    cfg_two = cf.cfg_two
    cfg_lm = cf.cfg_landmark
    rootDir = cf.root_testDir
    gt_json_dir = '/root/dataset_clp/dataset_v2/test/test.json'
    pd_output_json = '/root/deIdentification-clp/result/one_stage_result/test_bbox.json'
    
    # # YOLO를 사용한 Detection
    # result_one = ensemble.detection_result(config=cfg_one['configs'], checkpoint=cfg_one['checkpoints'], 
    #                           data_path=cfg_one['input_img'], save_dir=cfg_one['output_lbl'], iou_thr=0.5, imgsz=1280)
    
    # Using Pycoco
    pd_path = os.path.abspath(os.path.join(cfg_one['output_lbl'], os.pardir)) + "/result.json" 
    custom_to_coco_result(gt_json_dir, pd_path, pd_output_json)
    
    if not os.path.exists(pd_output_json) : 
        print(f"{pd_output_json} can not found")
        exit()
    else :
        cocoGt=COCO(gt_json_dir)
        cocoDt=cocoGt.loadRes(pd_output_json)
        imgIds=sorted(cocoGt.getImgIds())
        # imgIds=imgIds[0:100]
        print(f"Start COCO Evaluation.. Image Lenth : {len(imgIds)}")
        imgId = imgIds[np.random.randint(len(imgIds))]
        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
        cocoEval.params.maxDets = [300]
        print("License-Plate")
        cocoEval.params.catIds = [1]
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize_2()
        
        print("Vehicle")
        cocoEval.params.catIds = [2]
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize_2()

    # detections의 box좌표는 절대좌표 xyxy
    detections, groundtruths, classes = boundingBoxes(labelPath=cfg_one['input_lbl'],
                                                    predictPath=cfg_one['output_lbl'],
                                                    imagePath=cfg_one['input_img'])
    cal_mAP(cfg_en['num2class'], detections, groundtruths, classes, save_img=False, save_path='/root/deIdentification-clp/result/analysis_result/fp')
    
    if cfg_en['save_img'] :
        print(f"Save Result Images at {cfg_one['output_img']}...")
        os.makedirs(cfg_one['output_img'], exist_ok=True)
        boxPlot(detections + groundtruths, cfg_one['input_img'],
                savePath=cfg_one['output_img'])
        print(f"Finish to save result images at {cfg_one['output_img']}")
    