import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import CustomDataset as CDs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader


transform = Compose([
    Resize((224, 224)),
    ToTensor()
])
root_folder = 'nerf_synthetic/lego/'
dataset = CDs.CustomDataset(root_dir=root_folder, mode='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

def My_predict():
    X = []
    y = []
    for images, labels in dataloader:
        images = images.view(images.size(0), -1).numpy()
        labels = np.array([label[0][0].item() for label in labels])

        X.append(images)
        y.append(labels)

    X = np.vstack(X)
    y = np.hstack(y)
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    predictions = clf.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy}")

    sample_image = X[0].reshape(1, -1)
    predicted_label = clf.predict(sample_image)
    print(f"Predicted label: {predicted_label}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # def show_image(image_tensor, title):
    #     image = image_tensor.permute(1, 2, 0).numpy()
    #     plt.imshow(image)
    #     plt.title(title)
    #     plt.show()

    plt.figure(figsize=(10, 7))
    for i in range(len(X_pca)):
        if y[i] == predictions[i]:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], color='blue', label='Correct' if i == 0 else "")
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], color='red', label='Incorrect' if i == 0 else "")
    
    plt.title('SVM Classification Results')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    My_predict()
    