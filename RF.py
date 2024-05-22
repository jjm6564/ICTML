import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torchvision import models
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import CustomDataset as CDs

# Transform 정의
transform = Compose([
    Resize((224, 224)),  # 이미지 크기 조정
    ToTensor()  # 이미지를 텐서로 변환
])

# Dataset 및 DataLoader 생성
root_folder = 'nerf_synthetic/lego/'
dataset = CDs.CustomDataset(root_dir=root_folder, mode='val', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# CNN 모델을 사용하여 특징 추출
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, labs in dataloader:
            images = images.cuda()
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(labs.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

if __name__ == "__main__":
# 사전 학습된 ResNet50 모델을 사용하여 특징 추출
    cnn_model = models.resnet50(pretrained=True)
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])  # 마지막 FC layer 제거
    cnn_model = cnn_model.cuda()

    # 특징 추출
    features, labels = extract_features(cnn_model, dataloader)
    features = features.reshape(features.shape[0], -1)  # (N, 2048) 형태로 변경

    # 라벨을 1차원 벡터로 변환
    labels = labels[:, 0, 0]  # 필요한 경우 라벨 형태에 맞게 수정

    # 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Random Forest 모델 학습
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 테스트 데이터셋으로 예측
    y_pred = rf_model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # 결과 시각화 (예시)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], 'r', label='Actual')
    plt.plot(y_pred[:100], 'b', label='Predicted')
    plt.legend()
    plt.show()
