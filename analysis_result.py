import argparse
import csv
import json
import os
import ast
import cv2
from tqdm import tqdm
from weighted_box_fusion.utils import boundingBoxes, cal_mAP, boxPlot, convert_scaled_predict
import weighted_box_fusion.clp_ensemble as ensemble
import clp_landmark_detection.detect as landmark
from PIL import Image
import numpy as np
import config as cf
from deId import deIdentify

def parse_args():
    parser = argparse.ArgumentParser(description='DeIdentify Dataset')
    parser.add_argument(
        '--dataset', default="/root/dataset_clp/dataset_2044_new/test_only_one/", help='Test Dataset Path')
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
    # YOLO를 사용한 Detection
    result_one = ensemble.detection_result(config=cfg_one['configs'], checkpoint=cfg_one['checkpoints'], 
                              data_path=cfg_one['input_img'], save_dir=cfg_one['output_lbl'], iou_thr=0.5, imgsz=1280)
    # detections의 box좌표는 절대좌표 xyxy
    detections, groundtruths, classes = boundingBoxes(labelPath=cfg_one['input_lbl'],
                                                      predictPath=cfg_one['output_lbl'],
                                                      imagePath=cfg_one['input_img'])
    cal_mAP(cfg_en['num2class'], detections, groundtruths, classes, save_img=True, save_path='/root/deIdentification-clp/result/analysis_result/fp')
    
    if cfg_en['save_img'] :
        print(f"Save Result Images at {cfg_one['output_img']}...")
        os.makedirs(cfg_one['output_img'], exist_ok=True)
        boxPlot(detections + groundtruths, cfg_one['input_img'],
                savePath=cfg_one['output_img'])
        print(f"Finish to save result images at {cfg_one['output_img']}")
    