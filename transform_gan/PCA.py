import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
from torch.autograd import Variable
from tqdm import tqdm
import fnmatch

# ResNet 모델을 사용하여 이미지 특성을 추출하는 함수

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

def extract_features(image_path):
    # 사전 훈련된 ResNet 모델 로드
    resnet_model = models.resnet18(pretrained=True)

    # 모델을 평가 모드로 설정
    resnet_model.eval()

    # 이미지 불러오기 및 전처리
    img = Image.open(image_path)
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    img = preprocess(img)
    img = img.unsqueeze(0)  # 배치 차원을 추가

    # 이미지를 모델의 입력 형식에 맞게 변환
    img = Variable(img)

    # 특성 추출
    features = resnet_model(img)

    return features


# 가상 이미지 데이터와 실제 이미지 데이터 디렉토리 경로
# virtual_data_dir = "/root/deIdentification-clp/cross_analysis/test/virtual"
# real_data_dir = "/root/deIdentification-clp/cross_analysis/test/real"

virtual_data_dir = "/root/deid-lp-GAN/results/license-plate_cyclegan_AtoB/license-plate_cyclegan/test_latest/images"

# a_b_list = read_file_real_A_B(os.path.join(current_dir, virtual_data_dir))
# avg_psnr, avg_ssim, avg_mse = calculate_psnr_ssim(a_b_list, os.path.join(current_dir, dir_path))

# 이미지 특성 벡터와 라벨 초기화
data = []
labels = []

# 가상 이미지 데이터 처리
for filename in tqdm(os.listdir(virtual_data_dir)):
    if filename.endswith(".png") and "real_A" in filename:
        image_path = os.path.join(virtual_data_dir, filename)
        features = extract_features(image_path)
        features = features.detach().numpy()
        features = features[0]
        data.append(features)
        labels.append("real_A")
    if filename.endswith(".png") and "real_B" in filename:
        image_path = os.path.join(virtual_data_dir, filename)
        features = extract_features(image_path)
        features = features.detach().numpy()
        features = features[0]
        data.append(features)
        labels.append("real_B")


# PCA를 사용한 차원 감소
data = np.array(data)
pca = PCA(n_components=2)
embedded_data = pca.fit_transform(data)

virtual_data = []
real_data = []

for idx, lbl in enumerate(labels) :
    if lbl == "real_A" :
        virtual_data.append(embedded_data[idx])
    else :
        real_data.append(embedded_data[idx])

virtual_data = np.array(virtual_data)
real_data = np.array(real_data)
# 데이터 시각화
plt.scatter(virtual_data[:, 0], virtual_data[:, 1], label="real_A")
plt.scatter(real_data[:, 0], real_data[:, 1], label="real_B")
plt.xlabel("First Principal Component")  # X축 이름 추가
plt.ylabel("Second Principal Component") # Y축 이름 추가
plt.legend()
plt.title("PCA Visualization")
save_dir = '/root/deid-lp-GAN/data_analysis_plot/pca_plot'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir,'result_realA_realB.jpg'))
