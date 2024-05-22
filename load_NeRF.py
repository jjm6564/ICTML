import test as ts
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

def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def train_nerf():
    root_folder = 'nerf_synthetic/lego/'
    transform = Compose([Resize((224, 224)), ToTensor()])
    train_dataset = ts.CustomDataset(root_dir=root_folder, mode='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    model = ts.NeRF().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_dataloader:
            images = images.cuda()
            labels = labels.squeeze(0).cuda()

            H, W = images.shape[2:]
            focal = 1.0
            rays_o, rays_d = get_rays(H, W, focal, labels.cpu().numpy())

            rays_o = torch.tensor(rays_o, dtype=torch.float32).view(-1, 3).cuda()
            rays_d = torch.tensor(rays_d, dtype=torch.float32).view(-1, 3).cuda()

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
    visualize_3d_results(model, train_dataloader, root_folder, transform)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {checkpoint_path}. Epoch {epoch}, Loss: {loss}")
        return model, optimizer, epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")

def visualize_3d_results(model, dataloader, root_folder, transform):
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.squeeze(0).cuda()

            H, W = images.shape[2:]
            focal = 1.0
            rays_o, rays_d = get_rays(H, W, focal, labels.cpu().numpy())

            rays_o = torch.tensor(rays_o, dtype=torch.float32).view(-1, 3).cuda()
            rays_d = torch.tensor(rays_d, dtype=torch.float32).view(-1, 3).cuda()

            outputs = model(rays_o, rays_d)
            rgb_outputs = outputs[..., :3].view(H, W, 3).cpu().numpy()
            alpha_outputs = outputs[..., 3].view(H, W).cpu().numpy()

            break

    # 3D 재구성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Reconstruction")

    # 포인트 클라우드 생성
    x = rays_o[:, 0].detach().cpu().numpy()
    y = rays_o[:, 1].cpu().numpy()
    z = rays_o[:, 2].cpu().numpy()
    c = rgb_outputs.reshape(-1, 3)

    ax.scatter(x, y, z, c=c, s=0.1)
    plt.show()

if __name__ == '__main__':
    train_nerf()
