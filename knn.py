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
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])
    cnn_model = cnn_model.cuda()

    features, labels = extract_features(cnn_model, dataloader)
    features = features.reshape(features.shape[0], -1)  

    labels = labels[:, 0, 0]  
    bins = np.linspace(min(labels), max(labels), 10)
    labels = np.digitize(labels, bins) - 1

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], 'r', label='Actual')
    plt.plot(y_pred[:100], 'b', label='Predicted')
    plt.legend()
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    Knn()