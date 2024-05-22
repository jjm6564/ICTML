import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
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
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labellist[idx])
        return image, label

# NeRF 모델 정의
class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)  # 3 for RGB, 1 for density

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        rgb = torch.sigmoid(x[:, :3])  # Ensure RGB values are in [0, 1]
        density = torch.relu(x[:, 3])  # Ensure density is positive
        return rgb, density

# 레이 생성 함수
def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)  # [H, W, 3]
    rays_d = np.dot(dirs, c2w[:3, :3].T)  # [H, W, 3]
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)  # [H, W, 3]
    return rays_o, rays_d

# JSON 파일에서 데이터 로드
def load_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    transform_matrices = [np.array(frame['transform_matrix']) for frame in frames]

    return transform_matrices

# 3D 객체 재구성 함수
def reconstruct_3d_object(nerf_model, points, transform_matrices, device):
    transformed_points = []
    for transform_matrix in transform_matrices:
        transformed_point = np.dot(points, np.transpose(transform_matrix[:3, :3])) + transform_matrix[:3, 3]
        transformed_points.append(transformed_point)
    transformed_points = np.concatenate(transformed_points, axis=0)
    points_tensor = torch.tensor(transformed_points, dtype=torch.float32).to(device)
    with torch.no_grad():
        rgb, density = nerf_model(points_tensor)
    rgb = rgb.cpu().numpy()
    density = density.cpu().numpy()
    density_threshold = 0.1
    sampled_points = transformed_points[density > density_threshold]
    sampled_rgb = rgb[density > density_threshold]
    return sampled_points, sampled_rgb

# 3D 출력을 위한 함수
def plot_3d_object(points, rgb):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb, s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('NeRF 3D Object')

    plt.show()

# 예측된 밀도에 따라 3D 객체를 생성하고 시각화하는 함수
def generate_and_visualize_object(nerf_model, points, transform_matrices, device):
    # Transform all points using the given transform_matrices
    transformed_points = []
    for transform_matrix in transform_matrices:
        # Transform points using the matrix
        transformed_point = np.dot(points, np.transpose(transform_matrix[:3, :3])) + transform_matrix[:3, 3]
        transformed_points.append(transformed_point)

    # Convert to numpy array
    transformed_points = np.concatenate(transformed_points, axis=0)

    points_tensor = torch.tensor(transformed_points, dtype=torch.float32).to(device)

    # Predict RGB and density
    with torch.no_grad():
        rgb, density = nerf_model(points_tensor)

    rgb = rgb.cpu().numpy()
    density = density.cpu().numpy()

    # Sample points based on density
    density_threshold = 0.1  # Adjust threshold as needed
    sampled_points = transformed_points[density > density_threshold]
    sampled_rgb = rgb[density > density_threshold]

    # Visualize sampled points
    plot_3d_object(sampled_points, sampled_rgb)

# Training (필요에 따라 구현)
def train_nerf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_folder = 'nerf_synthetic/lego/'
    transform = Compose([Resize((512, 512)), ToTensor()])
    train_dataset = CustomDataset(root_dir=root_folder, mode='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    num_epochs = 100  # 에포크 수를 크게 설정하여 충분히 학습
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.squeeze(0).to(device)

            H, W = images.shape[2:]
            focal = 1.0
            c2w = labels

            rays_o, rays_d = get_rays(H, W, focal, c2w.cpu().numpy())

            rays_o = torch.tensor(rays_o, dtype=torch.float32).view(-1, 3).to(device)
            rays_d = torch.tensor(rays_d, dtype=torch.float32).view(-1, 3).to(device)

            optimizer.zero_grad()
            outputs_rgb, outputs_density = model(rays_o)
            outputs = torch.cat([outputs_rgb, outputs_density.unsqueeze(-1)], dim=-1)

            images = images.permute(0, 2, 3, 1).view(-1, 3)
            if images.shape[0] > outputs.shape[0]:
                images = images[:outputs.shape[0], :]

            rgb_outputs = outputs[..., :3]

            loss = criterion(rgb_outputs, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}")

        # 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'nerf_model_epoch_{epoch+1}.pth')

    print("Training complete.")
    torch.save(model.state_dict(), 'nerf_model_final.pth')
    
def generate_data(num_points):
    points = np.random.randn(num_points, 3) *2-1 # Random 3D points
    target_rgb = np.random.rand(num_points, 3)  # Random RGB colors
    target_density = np.random.rand(num_points)  # Random densities
    return points, target_rgb, target_density
if __name__ == "__main__":
    # Replace with your JSON file path
    json_file = 'nerf_synthetic/lego/train.json'

    # Load data from JSON file
    transform_matrices = load_data_from_json(json_file)

    # Create NeRF model instance
    nerf_model = NeRF()

    # Load trained model if needed
    model_path = "nerf_model_final.pth"
    if os.path.exists(model_path):
        nerf_model.load_state_dict(torch.load(model_path))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    nerf_model.to(device)

    # Generate and visualize 3D object
    N = 1000  # Increase the number of points
    points, target_rgb, target_density = generate_data(N)

    generate_and_visualize_object(nerf_model, points, transform_matrices, device)
