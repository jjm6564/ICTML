import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import CustomDataset as CDs
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CTroot_folder = CDs.root_folder
CTtransform = CDs.transform

dataset = CDs.CustomDataset(root_dir=CTroot_folder, mode='test', transform=CTtransform)
dataloader = CDs.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


def My_predict():
    X = []
    y = []
    for images, labels in dataloader:
        images = images.view(images.size(0), -1).numpy()  # Flatten images and convert to numpy array
        labels = np.array([label[0][0].item() for label in labels])

        X.append(images)
        y.append(labels)

    X = np.vstack(X)
    y = np.hstack(y)
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    # SVM 학습
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(X, y)

    # 모델 평가
    predictions = clf.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy}")

    # 모델 예측 예제
    sample_image = X[0].reshape(1, -1)
    predicted_label = clf.predict(sample_image)
    print(f"Predicted label: {predicted_label}")


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)


    def show_image(image_tensor, title):
        image = image_tensor.permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.title(title)
        plt.show()

    # 첫 번째 배치의 이미지와 예측된 레이블 시각화
    # for i in range(1):  # 예시로 첫 5개 이미지를 출력합니다.
    #     show_image(dataset[i][0], f"True Label: {y[i]}")

    plt.figure(figsize=(10, 7))
    for i in range(len(X_pca)):
        if y[i] == predictions[i]:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], color='blue', label='Correct' if i == 0 else "")
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], color='red', label='Incorrect' if i == 0 else "")
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Classification Results')
    plt.legend()
    plt.show()


# 이미지 데이터 및 레이블 준비

if __name__ == '__main__':
    # for images, labels in dataloader:
    #     images = images.view(images.size(0), -1).numpy()  # Flatten images and convert to numpy array
    #     labels = np.array([label[0][0].item() for label in labels])

    #     X.append(images)
    #     y.append(labels)

    # X = np.vstack(X)
    # y = np.hstack(y)
    
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    # # SVM 학습
    # clf = svm.SVC(kernel='linear')
    # clf.fit(X, y)

    # # 모델 평가
    # predictions = clf.predict(X)
    # accuracy = accuracy_score(y, predictions)
    # print(f"Accuracy: {accuracy}")

    # # 모델 예측 예제
    # sample_image = X[0].reshape(1, -1)
    # predicted_label = clf.predict(sample_image)
    # print(f"Predicted label: {predicted_label}")

    # def show_image(image_tensor, title):
    #     image = image_tensor.permute(1, 2, 0).numpy()
    #     plt.imshow(image)
    #     plt.title(title)
    #     plt.show()

    # # 첫 번째 배치의 이미지와 예측된 레이블 시각화
    # for i in range(6):  # 예시로 첫 5개 이미지를 출력합니다.
    #     show_image(dataset[i][0], f"True Label: {y[i]}, Predicted: {predictions[i]}")
    My_predict()