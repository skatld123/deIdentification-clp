import fnmatch
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.manifold import TSNE
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

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


def filter_real_images(directory, filename1, filename2, filename3, num_pairs=None):
    all_files = sorted(os.listdir(directory))
    real_images = {}

    for file in all_files:
        # file = os.path.splitext(file)[0]
        spilter = None
        if filename1 in file :
            spilter = filename1
        elif filename2 in file :
            spilter = filename2
        elif filename3 in file :
            spilter = filename3
        if spilter is not None : 
            base_name = file.split("_"+spilter)[0] + ".png"
            if base_name not in real_images:
                real_images[base_name] = []
            real_images[base_name].append(file)

    valid_triplets = []
    for base_name, files in real_images.items():
        if len(files) == 3:
            valid_triplets.extend(files)
            if num_pairs and len(valid_triplets) // 3 >= num_pairs:
                valid_triplets = valid_triplets[:num_pairs * 3]
                break

    return valid_triplets

# 이미지 특성 추출 및 PCA 분석 함수


def tsne_analysis(file_list, img_dir, tsne, scaler):
    data = []
    labels = []

    for filename in tqdm(file_list):
        image_path = os.path.join(img_dir, filename)
        features = extract_features(image_path)
        features = features.detach().numpy()
        features = features[0]
        data.append(features)
        if filename1 in filename :
            spilter = filename1
            spilter = "Input"
        elif filename2 in filename :
            spilter = filename2
            spilter = "GT"
        elif filename3 in filename :
            spilter = filename3
            spilter = "Output"
        # fn, type, domainWithExt = filename.split('_')
        # labels.append(f'{type}_{os.path.splitext(domainWithExt)[0]}')
        labels.append(spilter)

    data = np.array(data)
    embedded_data = tsne.fit_transform(data)
    embedded_data = scaler.fit_transform(embedded_data)

    return embedded_data, labels


# 가상 이미지 데이터와 실제 이미지 데이터 디렉토리 경로
img_dir = "D://dataset_virtual//결과 데이터셋//license-plate_cyclegan_v2_no_aug_testst//test_latest//only_parkinglot"

filename1 = 'real_A'
filename2 = 'real_B'
filename3 = 'fake_B'
file_list = filter_real_images(img_dir, filename1, filename2, filename3, 10)

# PCA 및 스케일러 초기화
perplexitys = [20]
for perplexity in perplexitys :

    tsne = TSNE(n_components=3, random_state=0, perplexity=perplexity)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # 모든 데이터에 대한 PCA 분석
    embedded_data, labels = tsne_analysis(file_list, img_dir, tsne, scaler)

    # 각 조합에 대한 시각화
    combinations = [('Input', 'GT'), ('Output', 'GT'),
                    ('Input', 'GT', 'Output')]
    for combo in combinations:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3D 그래프 생성
        for label in combo:
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            if label == 'GT':
                marker = 's'
                color = 'forestgreen'
            elif label == 'Input':
                marker = 'o'
                color = 'firebrick'
            else:
                marker = 'x'
                color = 'orange'
            ax.scatter(embedded_data[indices, 0], embedded_data[indices, 1],
                    embedded_data[indices, 2], label=label, color=color, marker=marker)

        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        ax.set_zlabel("Third Principal Component")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.legend()
        plt.title(f"t-SNE Visualization ({' & '.join(combo)})")
        save_dir = 'C://Users//nseungho//Desktop//남승호_졸업관련//비식별화_번호판_tSNE_PCA 분석//result_1230'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'result_{"_".join(combo)}_{perplexity}.jpg'))
        plt.show()
