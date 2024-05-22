import os
import json
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        image = Image.open(self.imglist[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labellist[idx], dtype=torch.float32)
        return image, label

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i != 4 else nn.Linear(W + input_ch, W) for i in range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, input_pts, input_views):
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i == 4:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs

def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def train_nerf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_folder = 'nerf_synthetic/lego/'
    transform = Compose([Resize((512, 512)), ToTensor()])  # 이미지 크기를 더 크게 조정
    train_dataset = CustomDataset(root_dir=root_folder, mode='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.squeeze(0).to(device)

            H, W = images.shape[2:]
            focal = 1.0
            rays_o, rays_d = get_rays(H, W, focal, labels.cpu().numpy())

            rays_o = torch.tensor(rays_o, dtype=torch.float32).view(-1, 3).to(device)
            rays_d = torch.tensor(rays_d, dtype=torch.float32).view(-1, 3).to(device)

            optimizer.zero_grad()
            outputs = model(rays_o, rays_d)

            images = images.permute(0, 2, 3, 1).view(-1, 3)
            if images.shape[0] > outputs.shape[0]:
                images = images[:outputs.shape[0], :]

            rgb_outputs = outputs[..., :3]

            loss = criterion(rgb_outputs, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}")

    print("Training complete.")
    visualize_3d_results(model, train_dataloader, device)

def visualize_3d_results(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.squeeze(0).to(device)

            H, W = images.shape[2:]
            focal = 1.0
            rays_o, rays_d = get_rays(H, W, focal, labels.cpu().numpy())

            rays_o = torch.tensor(rays_o, dtype=torch.float32).view(-1, 3).to(device)
            rays_d = torch.tensor(rays_d, dtype=torch.float32).view(-1, 3).to(device)

            outputs = model(rays_o, rays_d)
            rgb_outputs = outputs[..., :3].view(H, W, 3).cpu().numpy()
            alpha_outputs = outputs[..., 3].view(H, W).cpu().numpy()

            break

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Reconstruction")

    x = rays_o[:, 0].cpu().numpy()
    y = rays_o[:, 1].cpu().numpy()
    z = rays_o[:, 2].cpu().numpy()
    c = rgb_outputs.reshape(-1, 3) / 255.0

    c = np.clip(c, 0, 1)

    ax.scatter(x, y, z, c=c, s=10)  # s 값을 조정하여 점의 크기를 키웁니다.
    ax.view_init(elev=30, azim=30)  # 시야각을 조정하여 이미지가 잘 보이도록 합니다.
    plt.show()

if __name__ == '__main__':
    train_nerf()
