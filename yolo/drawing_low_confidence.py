import os
import cv2
from PIL import Image

"""
    YOLO의 Detection 결과를 이용하여 전체 이미지에서 낮은 confidence score의 박스를 그리는 코드 
"""

# 이미지와 annotation 디렉토리 경로
image_dir = '/root/dataset_clp/dataset_2044_new/test/images/'
annotation_dir = '/usr/src/app/runs/detect/exp6/labels/'
output_dir = '/root/false_positive/'

conf_score = 0.3
# annotation 파일들을 순회하면서 바운딩 박스 그리기 및 저장
for annotation_filename in os.listdir(annotation_dir):
    if annotation_filename.endswith('.txt'):
        image_filename = os.path.splitext(annotation_filename)[0] + '.jpg'
        image_path = os.path.join(image_dir, image_filename)
        annotation_path = os.path.join(annotation_dir, annotation_filename)
        
        # 이미지 로드
        image = cv2.imread(image_path)
        
        # annotation 파일 읽기
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        # 각 줄에서 정보 추출하여 바운딩 박스 그리기
        isDetect = False
        for line in lines:
            class_id, x_center, y_center, width_rel, height_rel, confidence = map(float, line.strip().split())
            if confidence >  conf_score:  # Confidence score가 0.5 이하인 경우에만 그림
                img_height, img_width, _ = image.shape
                x = int((x_center - width_rel / 2) * img_width)
                y = int((y_center - height_rel / 2) * img_height)
                w = int(width_rel * img_width)
                h = int(height_rel * img_height)
                if class_id == 0 : 
                    color = (255, 0, 0)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)  # 바운딩 박스 그리기
                else :
                    color = (0, 255, 0)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)  # 바운딩 박스 그리기
                # Confidence score 표시
                text = f'{class_id} : {confidence:.2f}'
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)
                isDetect = True
        if isDetect : 
            output_dir += ("/bbox_result" + "_detect_" + str(conf_score))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            result_image_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(result_image_path, image)
            print(f'Processed and saved: {result_image_path}')
            output_dir = '/root/false_positive/'