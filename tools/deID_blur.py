import os
import cv2
import numpy as np

# 입력 디렉토리와 출력 디렉토리 설정
input_image_dir = '/root/dataset_clp/dataset_v2/test/images'
input_label_dir = '/root/dataset_clp/dataset_v2/test/labels'
output_dir = '/root/dataset_clp/dataset_v2/test/blur_images'

# 클래스 번호가 0인 객체에 대한 가우시안 블러 파라미터 설정
blur_kernel_size = (31, 31)  # 블러 커널 크기
blur_sigma = 80              # 블러 시그마 값

# 출력 디렉토리가 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 이미지 파일 목록 가져오기
image_files = os.listdir(input_image_dir)

for image_file in image_files:
    if image_file.endswith('.jpg'):
        # 이미지 파일 경로
        image_path = os.path.join(input_image_dir, image_file)

        # 레이블 파일 경로
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(input_label_dir, label_file)

        if os.path.exists(label_path):
            # 이미지 로드
            image = cv2.imread(image_path)

            # 레이블 파일 읽기
            with open(label_path, 'r') as label_file:
                for line in label_file:
                    line = line.strip().split()
                    class_id, x, y, w, h = map(float, line)
                    class_id = int(class_id)

                    # 클래스 번호가 0인 경우에만 가우시안 블러 적용
                    if class_id == 0:
                        # 바운딩 박스 좌표 계산
                        left = int((x - w / 2) * image.shape[1])
                        top = int((y - h / 2) * image.shape[0])
                        right = int((x + w / 2) * image.shape[1])
                        bottom = int((y + h / 2) * image.shape[0])

                        # 이미지에서 해당 영역에 가우시안 블러 적용
                        roi = image[top:bottom, left:right]
                        roi = cv2.GaussianBlur(roi, blur_kernel_size, blur_sigma)
                        image[top:bottom, left:right] = roi

            # 결과 이미지 저장
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, image)

print("가우시안 블러 적용 및 이미지 저장이 완료되었습니다.")
