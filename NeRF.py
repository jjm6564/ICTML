import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from PIL import Image
from CustomDataset import CustomDataset

class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        rgb = torch.sigmoid(x[:, :3])
        density = torch.relu(x[:, 3])
        return rgb, density

def plot_3d_object(points, rgb):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('NeRF 3D Object')
    plt.show()

def load_data_from_json(json_file, root_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    frames = data['frames']
    img_paths = []
    transform_matrices = []
    for frame in frames:
        img_path = os.path.join(root_dir, frame['file_path'][2:] + '.png')
        transform_matrix = np.array(frame['transform_matrix'])
        img_paths.append(img_path)
        transform_matrices.append(transform_matrix)
    return img_paths, transform_matrices

def generate_and_visualize_object(nerf_model, points, device):
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    with torch.no_grad():
        rgb, density = nerf_model(points_tensor)
    rgb = rgb.cpu().numpy()
    density = density.cpu().numpy()
    density_threshold = 0.5                                          #density
    sampled_points = points[density > density_threshold]
    sampled_rgb = rgb[density > density_threshold]
    plot_3d_object(sampled_points, sampled_rgb)

def train_nerf_from_dataset(root_folder, json_file, num_epochs=10):                     # Epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = Compose([Resize((224, 224)), ToTensor()])
    train_dataset = CustomDataset(root_folder, 'train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)         #Batch

    nerf_model = NeRF().to(device)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        nerf_model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            batch_size, channels, height, width = images.shape
            images = images.permute(0, 2, 3, 1).contiguous().view(-1, channels)[:, :3]

            rgb, density = nerf_model(images)
            target_rgb = images
            target_density = torch.ones(images.shape[0], dtype=torch.float32).to(device)

            loss_rgb = criterion(rgb, target_rgb)
            loss_density = criterion(density, target_density)
            loss = loss_rgb + loss_density
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        print(f"Epoch {epoch}, Average Loss: {total_loss / len(train_dataloader)}")

    torch.save(nerf_model.state_dict(), "nerf_model.pth")

    val_dataset = CustomDataset(root_folder, 'val', transform=transform)
    val_points = []
    
    for images, labels in DataLoader(val_dataset, batch_size=1):
        images = images.to(device)
        batch_size, channels, height, width = images.shape
        images = images.permute(0, 2, 3, 1).contiguous().view(-1, channels)[:, :3].cpu().numpy()
        val_points.append(images)
    val_points = np.concatenate(val_points, axis=0)

    generate_and_visualize_object(nerf_model, val_points, device)

if __name__ == "__main__":
    root_folder = 'nerf_synthetic/lego/'
    json_file = os.path.join(root_folder, 'transforms_train.json')
    train_nerf_from_dataset(root_folder, json_file)
