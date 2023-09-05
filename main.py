from Object-Detection-WBF.utils import boundingBoxes, getArea, getUnionAreas, boxesIntersect, getIntersectionArea, iou, AP, mAP, boxPlot
import Object-Detection-WBF.clp_ensemble as ensemble

# Specify the path to model config and checkpoint file
config_file_1 = '/root/De-identification-CLP/weights/dino_2044_new_50/dino-5scale_swin-l_8xb2-36e_coco.py'
# config_file_2 = '/root/mmdetection/configs/clp/effb3_fpn_8xb4-crop896-1x_clp.py'
# config_file_3 = '/root/mmdetection/configs/clp/dcnv2_clp.py'
config_file_4 = 'yolo'
config_list = [config_file_1,config_file_4]

checkpoint_file_1 = '/root/De-identification-CLP/weights/dino_2044_new_50/best_coco_bbox_mAP_epoch_38.pth'
# checkpoint_file_2 = '/root/mmdetection/work_dirs/effb3_2044_200/epoch_100.pth'
# checkpoint_file_3 = '/root/mmdetection/work_dirs/dcnv2_2044_200/epoch_100.pth'
checkpoint_file_4 = '/root/De-identification-CLP/weights/yolov8/best_1280.pt'
checkpoint_file_list = [checkpoint_file_1,checkpoint_file_4]

test_img_prefix = '/root/dataset_clp/dataset_2044/test/images/'

dir_prefix = ''

final_list = []
wbf_list = []
img_boxes_list = []
img_score_list = []
img_labels_list = []

if __name__ == '__main__':
    
    weights = [1, 1]
    # config_list, checkpoint_file_list 모델의 결과를 추론 및 앙살블
    ensemble.ensemble_result(config_list, checkpoint_file_list, 0.5, 0.0001, 0.1, weights)
    # 앙상블한 모델의 결과 추론
    num2class = {"0.0" : "license-plate", "1.0" : "vehicle"}
    detections, groundtruths, classes = boundingBoxes(labelPath="/root/dataset_clp/dataset_2044_new/test/labels", 
                                                  predictPath="/root/De-identification-CLP/ensemble_model/result/predict", 
                                                  imagePath="/root/dataset_clp/dataset_2044_new/test/images")
    print(detections)
    print(groundtruths)
    print(classes)
    # 박스 저장은 안하기에 생략
    # boxPlot(detections, "image", savePath="boxed_images/detection")
    # boxPlot(groundtruths, "image", savePath="boxed_images/groundtruth")
    # boxPlot(detections + groundtruths, "/root/dataset_clp/dataset_2044_new/test/images", savePath="/root/De-identification-CLP/ensemble_model/test_img")
    # IoU
    boxA = detections[-1][-1]
    boxB = groundtruths[-1][-1]

    print(f"boxA coordinates : {(boxA)}")
    print(f"boxA area : {getArea(boxA)}")
    print(f"boxB coordinates : {(boxB)}")
    print(f"boxB area : {getArea(boxB)}")
    print(f"Union area of boxA and boxB : {getUnionAreas(boxA, boxB)}")
    print(f"Does boxes Intersect? : {boxesIntersect(boxA, boxB)}")
    print(f"Intersection area of boxA and boxB : {getIntersectionArea(boxA, boxB)}")
    print(f"IoU of boxA and boxB : {iou(boxA, boxB)}")

    result = AP(detections, groundtruths, classes)

    print(result)

    for r in result:
        print("{:^8} AP : {}".format(num2class[str(r['class'])], r['AP']))
    print("---------------------------")
    print(f"mAP : {mAP(result)}")
    
    # detections을 받아서 원본 이미지에서 크롭하기, 
    
    # 크롭된 이미지를 Plate-Landmarks-detection에 입력으로 넣기 
    # 이 때, 이미지 이름을 보존한 채로 Plate-Landmarks-detection의 포인트를 획득 한 뒤 해당 영역에 대한 전체적인 마스크를 그려볼 것
    
    # Plate-Landmarks-detection의 결과이미지를 받고 deidentification해서 번호판 붙이기
    
    # 잘랐던 바운딩 박스영역에 deid한 이미지를 붙이기