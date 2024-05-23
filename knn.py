import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torchvision import models
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import CustomDataset as CDs


transform = Compose([
    Resize((224, 224)),
    ToTensor()
])
root_folder = 'nerf_synthetic/lego/'
dataset = CDs.CustomDataset(root_dir=root_folder, mode='test', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

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

def Knn():
    cnn_model = models.resnet50(pretrained=True)
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])  # 마지막 FC layer 제거
    cnn_model = cnn_model.cuda()

    features, labels = extract_features(cnn_model, dataloader)
    features = features.reshape(features.shape[0], -1)  # (N, 2048) 형태로 변경

    # 라벨을 범주형으로 변환
    labels = labels[:, 0, 0]  # 필요한 경우 라벨 형태에 맞게 수정
    bins = np.linspace(min(labels), max(labels), 10)
    labels = np.digitize(labels, bins) - 1

    # 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # KNN 모델 학습
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # 모델 평가
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], 'r', label='Actual')
    plt.plot(y_pred[:100], 'b', label='Predicted')
    plt.legend()
    plt.show()

    #confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


# 사전 학습된 ResNet50 모델을 사용하여 특징 추출
if __name__ == "__main__":
    Knn()