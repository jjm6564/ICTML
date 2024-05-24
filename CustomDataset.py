from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import json

class CustomDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imglist, self.labellist = [], []
        self.mode = mode
        self.data_path = os.path.join(self.root_dir, self.mode)
        self.caption_path = os.path.join(self.root_dir, 'transforms_' + self.mode + '.json')

        with open(self.caption_path, 'r') as file:
            data = json.load(file)

        for frame in data['frames']:
            img_path = os.path.join(self.root_dir, frame['file_path'][2:] + '.png')
            self.imglist.append(img_path)
            self.labellist.append(frame['transform_matrix'])

        assert len(self.imglist) == len(self.labellist)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        image = Image.open(self.imglist[idx])
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labellist[idx])
        return image, label