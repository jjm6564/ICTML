import os
import json
from PIL import Image
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor


from sklearn import svm
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        """
        mode : train/val을 구분하기위해 편의상 만든 변수, 필요없다면 삭제 가능
        """

        self.root_dir = root_dir
        self.transform = transform
        self.imglist, self.labellist = [], []
        self.mode = mode
        """
        root_dir + 'train'/'val' 형태로 폴더가 저장되어 있을 경우 사용 가능
        label 이 json 파일에 저장되어 있는 경우
        """
        self.data_path = os.path.join(self.root_dir, self.mode)
        self.caption_path = os.path.join(self.root_dir, 'transforms_' + self.mode + '.json')

        with open(self.caption_path, 'r') as file:
            data = json.load(file)

        for frame in data['frames']:
            img_path = os.path.join(self.root_dir, frame['file_path'][2:] + '.png')  # "./train/r_0" -> "train/r_0.png"
            self.imglist.append(img_path)
            self.labellist.append(frame['transform_matrix'])

        assert len(self.imglist) == len(self.labellist)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        image = Image.open(self.imglist[idx])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labellist[idx])
        return image, label

root_folder = 'nerf_synthetic/lego/'

transform = Compose([
    Resize((224, 224)),  # resize image
    ToTensor()  # tranform image to tensor
])

# dataset = CustomDataset(root_dir=root_folder,
#                         mode='train',
#                         transform=transform)

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# if __name__ == '__main__':
# # Example of accessing the dataloader
#     for images, labels in dataloader:
#         # print(f"Batch of images shape: {images.shape}")
#         # print(f"Batch of labels: {labels.shape}")
    
#         #train_features, train_labels = next(iter(dataloader))
#         print(f"Feature batch shape: {images.size()}")
#         print(f"Labels batch shape: {labels.size()}")
#         img = images[0].squeeze()
#         label = labels[0]

#         if img.shape[0] == 4:  
#             img = img[:3, ...]
#         img = img.permute(1, 2, 0)
#         plt.imshow(img)
#         plt.show()
#         print(f"Label: {label}")