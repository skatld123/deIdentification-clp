import argparse
import csv
import json
import os
import ast
import cv2
from tqdm import tqdm
from weighted_box_fusion.utils import boundingBoxes, cal_mAP, boxPlot, convert_scaled_predict, boundingBoxes_fromDic
import weighted_box_fusion.clp_ensemble as ensemble
import clp_landmark_detection.detect as landmark
from PIL import Image
import numpy as npcond
import config as cf
from deId import deIdentify_blur_or_mask, deIdentify
from analysis_result import coco_evaluate_offline
from utils.crop import cropping_image_from_array, convert_row_from_csv

def parse_args():
    parser = argparse.ArgumentParser(description='DeIdentify Dataset')
    parser.add_argument(
        '--dataset', default="/root/dataset_clp/dataset_v2/test_only_one/", help='Test Dataset Path')
    parser.add_argument('--output', default="/root/deIdentification-clp/result/de_id_result", help='DeID output Directory')
    parser.add_argument('--gpu', default=0, help='Avaliable GPU Node')
    args = parser.parse_args()
    return args

def point_local2global(bbox_with_point, detection_result, save_dir) :
    '''
    bbox_with_point : 박스와 좌표가 포함되어 있는 dictionary
    detection_result : 검출 결과
    save_dir : global 좌표로 변환된 dic을 저장할 json_path
    '''
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
                detection_result_cropImg = detection_result[os.path.splitext(subImgName)[0]]
            else :
                print(f"no detection in {os.path.splitext(subImgName)[0]}")
                continue
            boxes, scores, labels = detection_result_cropImg['boxes'], detection_result_cropImg['scores'], detection_result_cropImg['labels']
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
        #         cv2.rectangle(img, (mx1, my1), (mx2, my2), (255,0,0), 4)
        # cv2.imwrite(f"/root/deIdentification-clp/result/{key}.jpg", img)
    with open(save_dir, 'w') as json_file:
        json.dump(global_dic, json_file)
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

    # YOLO를 사용한 Detection
    result_one = ensemble.detection_result(config=cfg_one['configs'], 
                                        checkpoint=cfg_one['checkpoints'], 
                                        data_path=cfg_one['input_img'], 
                                        save_dir=cfg_one['output_lbl'], 
                                        save_json_dir=cfg_one['output_json'],
                                        iou_thr=0.5, 
                                        imgsz=1280)
    # detections의 box좌표는 절대좌표 xyxy
    detections, groundtruths, classes = boundingBoxes(labelPath=cfg_one['input_lbl'],
                                                      predictPath=cfg_one['output_lbl'],
                                                      imagePath=cfg_one['input_img'])
    if cfg_one['save_img'] :
        print(f"Save Result Images at {cfg_one['output_img']}...")
        os.makedirs(cfg_one['output_img'], exist_ok=True)
        boxPlot(detections + groundtruths, cfg_one['input_img'],
                savePath=cfg_one['output_img'])
        print(f"Finish to save result images at {cfg_one['output_img']}")
    
    # detections을 받아서 원본 이미지에서 크롭하기, 
    # boxinfo = [filename, cls, conf, (x1, y1, x2, y2)]
    # 크롭 이미지와 그에 대한 좌표값이 필요 -> 그 좌표값을 전체 이미지에서의 좌표값을호 변환필욧
    dic_bbox_with_point = cropping_image_from_array(input_dir=cf.cfg_crop['input'], output_dir=cf.cfg_crop['output'], detections=detections, cls=1, save_img=True)
    
    box_point_json_path = os.path.join(cf.cfg_crop['output_dir'], 'box_with_point.json')
    with open(box_point_json_path, 'w') as json_file:
        json.dump(dic_bbox_with_point, json_file, indent=4)
    
    result_two = ensemble.detection_result(config=cfg_two['configs'], 
                                           checkpoint=cfg_two['checkpoints'], 
                                            data_path=cfg_two['input_img'], 
                                            save_dir=cfg_two['output_lbl'], 
                                            save_json_dir=cfg_two['output_json'], 
                                            iou_thr=0.3,
                                            imgsz=640)
    
    # result_two를 dic_bbox_with_point를 활용하여 좌표를 변환해줘야함s
    result_two = point_local2global(dic_bbox_with_point, result_two, save_dir=cfg_two['output_json'])
    
    detections, groundtruths, classes = boundingBoxes_fromDic(labelPath=cfg_one['input_lbl'],
                                                    prediction=result_two,
                                                    imgPath=cfg_one['input_img'])
    groundtruths_lp = [item for item in groundtruths if item[1] == 0.0]
    cal_mAP(cfg_two['num2class'], detections, groundtruths_lp, [0.0])
    
    if cfg_two['save_img'] :
        print(f"Save Result Images at {cfg_two['output_img']}...")
        os.makedirs(cfg_two['output_img'], exist_ok=True)
        boxPlot(boxlist=detections + groundtruths_lp, imagePath=cfg_one['input_img'],
                savePath=cfg_two['output_img'])
        print(f"Finish to save result images at {cfg_two['output_img']}")
        
    
    # if result_two is None or result_one is None : 
    with open('/root/deIdentification-clp/result/one_stage_result/result.json', "r") as json_file:
        result_one = json.load(json_file)
    
    with open('/root/deIdentification-clp/result/two_stage_result/result.json', "r") as json_file:
        result_two = json.load(json_file)
    
    # 앙상블 (relative xyxy로 나옴) 결과 파일을 사용
    print("Start Ensemble One-stage and Two-stage results")
    result_ensemble = ensemble.ensemble_result(result_one, result_two, save_json_dir=cfg_en['output_json'], save_dir=cfg_en['output_lbl'])
    
    detections, groundtruths, classes = boundingBoxes(labelPath=cfg_en['input_lbl'],
                                                      predictPath=cfg_en['output_lbl'],
                                                      imagePath=cfg_en['input_img'])
    
    print("AP of ENSEMBLE")
    cal_mAP(cfg_en['num2class'], detections, groundtruths, classes)
    result_json_path = cf.two_stage_output + "/result_coco.json"
    coco_evaluate_offline(gt_path=cfg_en['input_json'], pd_path=cfg_en['output_json'], output_path=result_json_path)
    
    # CSV 파일로 데이터 저장
    with open('ensemble_data.csv', 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(detections)

    # 차량 + 번호판의 바운딩박스 개수 1006개
    with open('ensemble_data.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        detections = [convert_row_from_csv(row) for row in csvreader]
    
    if cfg_en['save_img'] :
        print(f"Save Result Images at {cfg_en['output_img']}...")
        os.makedirs(cfg_en['output_img'], exist_ok=True)
        boxPlot(detections + groundtruths, cfg_en['input_img'],
                savePath=cfg_en['output_img'])
        print(f"Finish to save result images at {cfg_en['output_img']}")
        
    dic_bbox_with_point = cropping_image_from_array(input_dir=cf.cfg_crop_lp['input'], output_dir=cf.cfg_crop_lp['output'], detections=detections, cls=0, save_img=True)
    
    print("Start Landmark Detection...")
    input_cropping = cfg_lm['input_dir']
    print(f'Input Landmark Detection Dataset Size : {len(os.listdir(input_cropping))}')
    # filename, score, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    landmark_result = landmark.predict(backbone=cfg_lm['backbone'], checkpoint_model=cfg_lm['checkpoint'],
                                       save_img=True, save_txt=False,
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
    
    dic_predict = point_local2global(dic_bbox_with_point)
    
    # 잘랐던 바운딩 박스영역에 deid한 이미지를 붙이기
    deIdentify(dic_bbox_with_point, cfg_en['input_img'], cfg_lm['input_dir'], deid_output)
