import CustomDataset as CDs
import torch
import torch.nn as nn
import torch.optim as optim

CTroot_folder = CDs.root_folder
CTmode = 'train'
CTtransform = CDs.transform

class LinearSVM(nn.Module):
    def __init__(self, n_features):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

def svm_loss(outputs, labels):
    # SVM의 힌지 손실 구현
    return torch.sum(torch.clamp(1 - outputs.t() * labels, min=0))
# 데이터셋과 데이터로더 준비
dataset = CDs.CustomDataset(CTroot_folder,mode=CTmode,transform=CTtransform)
dataloader = CDs.DataLoader(dataset, batch_size=10, shuffle=True)

# 모델, 최적화 알고리즘, 에포크 수 설정
#model = LinearSVM(n_features=10)
model = LinearSVM(n_features=224 * 224 * 3)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 훈련 시작
num_epochs = 20
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # SVM은 기본적으로 이진 분류를 위한 알고리즘이므로,
        # 레이블을 -1과 1로 변환해야 합니다.
        y_batch = y_batch.float() * 2 - 1
        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        loss = svm_loss(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
