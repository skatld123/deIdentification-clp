import argparse
import csv
import json
import os
import ast
import cv2
from tqdm import tqdm
from weighted_box_fusion.utils import boundingBoxes, cal_mAP, boxPlot
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

def cropping_image(input_dir, output_dir, detections) :
    dic_bbox_with_point = {}
    print("Cropping only LicensePlate Bounding Box from Input Image")
    index = 0
    for idx, detection in enumerate(tqdm(detections)):
        file_name, label, conf, box = detection[0], float(
            detection[1]), float(detection[2]), ast.literal_eval(detection[3])
        if label == 0:
            img_path = os.path.join(input_dir, file_name + ".jpg")
            img = Image.open(img_path)
            cropped_img = img.crop(box)
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

def point_local2global(result_landmark_detection) :
    dic_predict = result_landmark_detection
    keys_to_delete = []
    for main_img, sub_img in dic_predict.items() :
        box_list = sub_img['boxes']
        if 'key_points' not in sub_img :
            keys_to_delete.append(main_img)
            continue
        kp_list = sub_img['key_points']
        for box, kp in zip(box_list, kp_list) :
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            for pt in kp :
                pt[0] = pt[0] + x1
                pt[1] = pt[1] + x1
    for key in keys_to_delete :
        del dic_predict[key]
    return dic_predict
        

if __name__ == '__main__':
    args = parse_args()
    deid_output = args.output
    cf.root_testDir = args.dataset
    
    current_directory = os.getcwd()
    print("현재 작업 디렉토리:", current_directory)
    
    # De-ID Result
    cfg_en = cf.cfg_ensemble
    cfg_lm = cf.cfg_landmark
    rootDir = cf.root_testDir
    # config_list, checkpoint_file_list 모델의 결과를 추론 및 앙살블
    ensemble.ensemble_result(cfg_en['configs'], cfg_en['checkpoints'], data_path=cfg_en['input_img'],
                             save_dir=cfg_en['save_txt_dir'], 
                             iou_thr=cfg_en['iou_thr'], skip_box_thr=cfg_en['skip_box_thr'], sigma=cfg_en['sigma'], 
                             weights=cfg_en['weight'])
    # predictPath는 앙상블한 예측 결과가 저장됨
    detections, groundtruths, classes = boundingBoxes(labelPath=cfg_en['input_lbl'],
                                                      predictPath=cfg_en['save_txt_dir'],
                                                      imagePath=cfg_en['input_img'])
    cal_mAP(cfg_en['num2class'], detections, groundtruths, classes)
    # CSV 파일로 데이터 저장
    with open('ensemble_data.csv', 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(detections)
        
    if cfg_en['save_img'] :
        print(f"Save Result Images at {cfg_en['save_img_dir']}...")
        boxPlot(detections + groundtruths, cfg_en['input_img'],
                savePath=cfg_en['save_img_dir'])
        print(f"Finish to save result images at {cfg_en['save_img_dir']}")
    
    # 차량 + 번호판의 바운딩박스 개수 1006개
    with open('ensemble_data.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        detections = list(csvreader)
    
    # detections을 받아서 원본 이미지에서 크롭하기, 
    # boxinfo = [filename, cls, conf, (x1, y1, x2, y2)]
    dic_bbox_with_point = cropping_image(input_dir=cfg_en['input_img'], output_dir=cfg_lm['input_dir'], detections=detections)
    
    print("Start Landmark Detection...")
    input_cropping = cfg_lm['input_dir']
    print(f'Input Landmark Detection Dataset Size : {len(os.listdir(input_cropping))}')
    # filename, score, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    landmark_result = landmark.predict(backbone=cfg_lm['backbone'], checkpoint_model=cfg_lm['checkpoint'],
                                       save_img=cfg_lm['save_img'], save_txt=cfg_lm['save_txt'],
                                        input_path=cfg_lm['input_dir'],
                                        output_path=cfg_lm['output_dir'],
                                        nms_threshold=cfg_lm['nms_thr'], vis_thres=cfg_lm['vis_thres'], imgsz=cfg_lm['imgsz'])
    # make Dictionary
    file_with_keypoint = {}
    for point in landmark_result :
        file_name, _, bbox, four_point = point[0], point[1], point[2], point[3]
        org_img_name = file_name.split("_crop_")[0]
        if org_img_name in file_with_keypoint : 
            file_with_keypoint[org_img_name][file_name] = four_point
        else :
            file_with_keypoint[org_img_name] = {file_name: four_point}
    print("End Landmark Detection!")
    
    found_nothing = []
    found_nothing_cropped = []
    # dic_bbox_with_point : 원래 이미지, Crop한 이미지와 그에 대한 바운딩 박스 좌표가 들어있는 딕셔너리
    # file_with_keypoint : Landmark Detection을 통해 나온 결과 값들, 원래 이미지, Crop한 이미지와 그에 대한 좌표 존재
    # 둘의 원래 이미지 개수가 다르다면 Landmark Detection을 통해 나온 값이 없는 것
    for key in dic_bbox_with_point :
        sub_img_list = dic_bbox_with_point[key]['subimage']
        if key in file_with_keypoint :
            key_points_mapping = file_with_keypoint[key]
            for sub_img in sub_img_list :
                if sub_img in key_points_mapping :
                    key_points_list = key_points_mapping[sub_img]
                    if 'key_points' in dic_bbox_with_point[key] :
                        dic_bbox_with_point[key]['key_points'].append(key_points_list)
                    else : dic_bbox_with_point[key]['key_points'] = [key_points_list]
                else : found_nothing_cropped.append(sub_img)
        else :
            found_nothing.append(key)
            
    print(f"Detected Nothing from Image : {found_nothing}, Detected Nothing Image Count : {len(found_nothing)}")
    print(f"Detected Nothing from Cropped Image : {found_nothing_cropped}, Detected Nothing Image Count : {len(found_nothing_cropped)}")
    print(f'Result_Img_count after LandmarkDetection : {len(file_with_keypoint)}')
    
    with open(('result_landmark_with_box.json'), 'w+') as json_file:
        json.dump(dic_bbox_with_point, json_file)
    
    with open('result_landmark_with_box.json', 'r') as json_file:
        dic_bbox_with_point = json.load(json_file)
    
    # dic_predict = point_local2global(dic_bbox_with_point)
    
    # 잘랐던 바운딩 박스영역에 deid한 이미지를 붙이기
    deIdentify(dic_bbox_with_point, cfg_en['input_img'], cfg_lm['input_dir'], deid_output)

    