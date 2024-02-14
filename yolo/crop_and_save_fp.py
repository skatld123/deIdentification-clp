import os
from PIL import Image

"""
    YOLO의 Detection 결과를 이용하여 False Positive를 자르고 저장하는 코드 
"""

# 이미지와 annotation 디렉토리 경로
image_dir = '/root/dataset_clp/dataset_2044_new/test/images/'
annotation_dir = '/usr/src/app/runs/val/exp6_0.5/labels/'
output_dir = '/root/false_positive/'

# annotation 파일 읽어서 크롭 수행
for annotation_filename in os.listdir(annotation_dir):
    annotation_path = os.path.join(annotation_dir, annotation_filename)
    image_filename, _ = os.path.splitext(annotation_filename)
    image_path = os.path.join(image_dir, image_filename + '.jpg')  # 이미지 확장자에 맞게 변경
    threshold = 0.5
    if os.path.isfile(annotation_path) and os.path.isfile(image_path):
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) == 6:
                    class_idx, rel_x, rel_y, rel_width, rel_height, confidence = map(float, parts)
                    if confidence <= threshold:
                        img = Image.open(image_path)
                        img_width, img_height = img.size
                        abs_x = int(rel_x * img_width)
                        abs_y = int(rel_y * img_height)
                        abs_width = int(rel_width * img_width)
                        abs_height = int(rel_height * img_height)
                        x = abs_x - abs_width // 2
                        y = abs_y - abs_height // 2
                        box = (x, y, x + abs_width, y + abs_height)
                        cropped_img = img.crop(box)
                        
                        # 결과 이미지 저장
                        output_filename = f'{image_filename}_{class_idx}_{confidence:.2f}.jpg'
                        if class_idx == 0.0 : 
                            output_dir += ("/license-plate" + "_" + str(threshold))
                        else : output_dir += ("/vehicle" + "_" + str(threshold))
                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_path = os.path.join(output_dir, output_filename)
                        cropped_img.save(output_path)
                        output_dir = '/root/false_positive/'
                        print(f'Saved cropped image: {output_filename}')
