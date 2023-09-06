import csv
import json
import os
import ast
from tqdm import tqdm
from weighted_box_fusion.utils import boundingBoxes, getArea, getUnionAreas, boxesIntersect, getIntersectionArea, iou, AP, mAP, boxPlot
import weighted_box_fusion.clp_ensemble as ensemble
import clp_landmark_detection.detect as landmark
from PIL import Image

# Specify the path to model config and checkpoint file
config_file_1 = '/root/deIdentification-clp/weights/dino_2044_new_50/dino-5scale_swin-l_8xb2-36e_coco.py'
# config_file_2 = '/root/mmdetection/configs/clp/effb3_fpn_8xb4-crop896-1x_clp.py'
# config_file_3 = '/root/mmdetection/configs/clp/dcnv2_clp.py'
config_file_4 = 'yolo'
config_list = [config_file_1, config_file_4]

checkpoint_file_1 = '/root/deIdentification-clp/weights/dino_2044_new_50/best_coco_bbox_mAP_epoch_38.pth'
# checkpoint_file_2 = '/root/mmdetection/work_dirs/effb3_2044_200/epoch_100.pth'
# checkpoint_file_3 = '/root/mmdetection/work_dirs/dcnv2_2044_200/epoch_100.pth'
checkpoint_file_4 = '/root/deIdentification-clp/weights/dino_2044_new_50/yolov8/best_1280.pt'
checkpoint_file_list = [checkpoint_file_1, checkpoint_file_4]

dir_prefix = ''

final_list = []
wbf_list = []
img_boxes_list = []
img_score_list = []
img_labels_list = []

if __name__ == '__main__':
    data_path = '/root/dataset_clp/dataset_2044/test/images/'
    save_dir = '/root/deIdentification-clp/weighted_box_fusion/result/'
    crop_save_dir = '/root/deIdentification-clp/clp_landmark_detection/data/dataset/test/'
    
    # weights = [1, 1]
    # # config_list, checkpoint_file_list 모델의 결과를 추론 및 앙살블
    # ensemble.ensemble_result(config_list, checkpoint_file_list, data_path=data_path,
    #                          save_dir=save_dir, 
    #                          iou_thr=0.5, skip_box_thr=0.0001, sigma=0.1, 
    #                          weights=weights)
    # # # 앙상블한 모델의 결과 추론
    # num2class = {"0.0" : "license-plate", "1.0" : "vehicle"}
    # # predictPath는 앙상블한 예측 결과가 저장됨
    # detections, groundtruths, classes = boundingBoxes(labelPath="/root/dataset_clp/dataset_2044_new/test/labels", 
    #                                               predictPath="/root/deIdentification-clp/weighted_box_fusion/result/predict", 
    #                                               imagePath="/root/dataset_clp/dataset_2044_new/test/images")
    # print(detections)
    # # CSV 파일로 데이터 저장
    # with open('data.csv', 'w+', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(detections)
    # print(groundtruths)
    # print(classes)
    # # 박스 저장은 안하기에 생략
    # # boxPlot(detections, "image", savePath="boxed_images/detection")
    # # boxPlot(groundtruths, "image", savePath="boxed_images/groundtruth")
    # # boxPlot(detections + groundtruths, "/root/dataset_clp/dataset_2044_new/test/images", 
    # #         savePath="/root/deIdentification-clp/result/ensemble_result_images")
    # # IoU
    # boxA = detections[-1][-1]
    # boxB = groundtruths[-1][-1]

    # print(f"boxA coordinates : {(boxA)}")
    # print(f"boxA area : {getArea(boxA)}")
    # print(f"boxB coordinates : {(boxB)}")
    # print(f"boxB area : {getArea(boxB)}")
    # print(f"Union area of boxA and boxB : {getUnionAreas(boxA, boxB)}")
    # print(f"Does boxes Intersect? : {boxesIntersect(boxA, boxB)}")
    # print(f"Intersection area of boxA and boxB : {getIntersectionArea(boxA, boxB)}")
    # print(f"IoU of boxA and boxB : {iou(boxA, boxB)}")

    # result = AP(detections, groundtruths, classes)

    # # print(result)

    # for r in result:
    #     print("{:^8} AP : {}".format(num2class[str(r['class'])], r['AP']))
    # print("---------------------------")
    # print(f"mAP : {mAP(result)}")
    
    # 차량 + 번호판의 바운딩박스 개수 1006개
    with open('data.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        detections = list(csvreader)
    
    # detections을 받아서 원본 이미지에서 크롭하기, 
    # boxinfo = [filename, cls, conf, (x1, y1, x2, y2)]
    dic_bbox_with_point = {}
    print("Cropping only LicensePlate Bounding Box from Input Image")
    for idx, detection in enumerate(tqdm(detections)) : 
        file_name, label, conf, box = detection[0], float(detection[1]), float(detection[2]), ast.literal_eval(detection[3])
        if label == 0 :
            img_path = os.path.join(data_path, file_name + ".jpg")
            img = Image.open(img_path)
            cropped_img = img.crop(box)
            cropped_img.save(f'{crop_save_dir}{file_name}_crop_{idx:04d}.jpg')
            if file_name in dic_bbox_with_point :
                dic_bbox_with_point[file_name]['subimage'].append(f'{file_name}_crop_{idx:04d}.jpg')
                dic_bbox_with_point[file_name]['boxes'].append(box)
            else : 
                dic_bbox_with_point[file_name] = {'subimage': [
                    f'{file_name}_crop_{idx:04d}.jpg'], 'boxes': [box]}
        else : continue
    print(f"Cropped result images saved at : {crop_save_dir} \nCropped result images cnt : {len(os.listdir(crop_save_dir))}")
    
    landmark_output_path = '/root/deIdentification-clp/clp_landmark_detection/results'
    checkpoint_model = '/root/deIdentification-clp/clp_landmark_detection/weights/Resnet50_Final.pth'
    landmark_backbone = 'resnet50'  # or mobile0.25
    
    print("Start Landmark Detection...")
    print(f'Input Landmark Detection Dataset Size : {len(os.listdir(crop_save_dir))}')
    # 크롭된 이미지를 Plate-Landmarks-detection에 입력으로 넣기 
    # filename, score, (bx1,by1, bx2,by2), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    landmark_result = landmark.predict(backbone=landmark_backbone, checkpoint_model=checkpoint_model,
                                        save_img=True, save_txt=True,
                                        input_path=crop_save_dir,
                                        output_path=landmark_output_path,
                                        nms_threshold=0.3, vis_thres=0.5,imgsz=320)
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
    with open(('result_landmark_with_box.json'), 'w') as json_file:
        json.dump(dic_bbox_with_point, json_file)
    
    # 이 때, 이미지 이름을 보존한 채로 Plate-Landmarks-detection의 포인트를 획득 한 뒤 해당 영역에 대한 전체적인 마스크를 그려볼 것
    
    # Plate-Landmarks-detection의 결과이미지를 받고 deidentification해서 번호판 붙이기
    
    # 잘랐던 바운딩 박스영역에 deid한 이미지를 붙이기