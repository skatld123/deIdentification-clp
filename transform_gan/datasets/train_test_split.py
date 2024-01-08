import os
import shutil
from sklearn.model_selection import train_test_split

def split_images(a_directory, b_directory):
    # A 디렉토리에서 이미지 파일 목록 가져오기
    a_images = [f for f in os.listdir(a_directory) if f.endswith('.jpg')]
    
    # 이미지 파일을 train, valid, test로 나누기 (7:2:1 비율)
    train_files, valid_test_files = train_test_split(a_images, test_size=0.3, random_state=42, shuffle=True)
    valid_files, test_files = train_test_split(valid_test_files, test_size=2/3, random_state=42, shuffle=True)
    
    # 파일 이동 함수
    def move_files(files, src_a, src_b, dest_a, dest_b):
        for file in files:
            shutil.move(os.path.join(src_a, file), os.path.join(dest_a, file))
            shutil.move(os.path.join(src_b, file), os.path.join(dest_b, file))
    trainA = '/root/deid-lp-GAN/datasets/license-plate_test_no_aug/trainA'
    trainB = '/root/deid-lp-GAN/datasets/license-plate_test_no_aug/trainB'
    valA = '/root/deid-lp-GAN/datasets/license-plate_test_no_aug/valA'
    valB = '/root/deid-lp-GAN/datasets/license-plate_test_no_aug/valB'
    testA = '/root/deid-lp-GAN/datasets/license-plate_test_no_aug/testA'
    testB = '/root/deid-lp-GAN/datasets/license-plate_test_no_aug/testB'
    os.makedirs(trainA, exist_ok=True)
    os.makedirs(trainB, exist_ok=True)
    os.makedirs(valA, exist_ok=True)
    os.makedirs(valB, exist_ok=True)
    os.makedirs(testA, exist_ok=True)
    os.makedirs(testB, exist_ok=True)
    # 파일 이동
    move_files(train_files, a_directory, b_directory, trainA, trainB)
    move_files(valid_files, a_directory, b_directory, valA, valB)
    move_files(test_files, a_directory, b_directory, testA, testB)

# 디렉토리 A와 B의 경로 설정
a_directory = "/root/dataset_clp/dataset_virtual_parkinglot/fake"
b_directory = "/root/dataset_clp/dataset_virtual_parkinglot/real"

# 함수 호출
split_images(a_directory, b_directory)
