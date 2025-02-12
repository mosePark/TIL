'''
3000장 autoaug 진행 후 학습에 넣기
'''
#%%
import os
import glob
import time
import random
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm

# 재현성을 위한 seed 설정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 추가적으로, CuDNN의 비결정적 알고리즘 사용을 방지
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seed 값 설정
set_seed(123)

#%%
# 사용자 정의 데이터셋 클래스 (폴더 내 모든 GIF 파일을 불러옴)
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        # data가 리스트라면, 그대로 파일 경로 리스트로 사용
        if isinstance(data, list):
            self.file_paths = data
        else:
            # data가 문자열 (폴더 경로)라면, 해당 폴더 내의 *.gif 파일들을 불러옴
            self.file_paths = glob.glob(os.path.join(data, "*.gif"))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        
        filename = os.path.basename(path).lower()
        if "key" in filename:
            label = 0
        elif "watch" in filename:
            label = 1
        else:
            raise ValueError("Unknown label in file: " + filename)
        return image, label

# 폴더 경로 설정
train_folder     = 'C:/Users/UOS/Desktop/begas/data/B/EX2/train'
train_aug_folder = 'C:/Users/UOS/Desktop/begas/data/B/EX2/train_aug'
test_folder      = 'C:/Users/UOS/Desktop/begas/data/B/EX2/test'

# 각 폴더에서 GIF 파일 경로들을 가져옴
train_files     = glob.glob(os.path.join(train_folder, "*.gif"))
train_aug_files = glob.glob(os.path.join(train_aug_folder, "*.gif"))

# 두 폴더의 이미지 경로를 합침
combined_files = train_files + train_aug_files
print("총 이미지 수:", len(combined_files))  # 예: 3030개

# seed 설정 후 random.shuffle을 통해 재현 가능한 무작위 분할
random.shuffle(combined_files)
split_idx = int(0.8 * len(combined_files))
train_file_paths = combined_files[:split_idx]
val_file_paths   = combined_files[split_idx:]
print("학습 이미지 수:", len(train_file_paths))
print("검증 이미지 수:", len(val_file_paths))

# 변환(Transform) 정의 (이미지 크기를 128x128로 조정하고 Tensor로 변환)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 데이터셋 생성
train_dataset = CustomDataset(train_file_paths, transform=train_transform)
val_dataset   = CustomDataset(val_file_paths, transform=val_transform)
test_files    = glob.glob(os.path.join(test_folder, "*.gif"))
test_dataset  = CustomDataset(test_files, transform=test_transform)

# DataLoader 생성
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# CNN 모델 정의
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 128, 16, 16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 디바이스 설정, 모델, 손실 함수, 옵티마이저 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 및 검증 손실 기록을 위한 리스트
train_losses = []
val_losses = []

num_epochs = 10
total_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    # 학습 진행 (tqdm 사용)
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # 검증 평가 및 손실 계산
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_val = criterion(outputs, labels)
            running_val_loss += loss_val.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    val_acc = correct / total
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_duration:.2f} sec")

total_duration = time.time() - total_start_time
print("학습 완료 | 총 소요 시간: {:.2f} sec".format(total_duration))


#%%
# 테스트 셋에 대해 평가 (테스트 셋은 8장)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
test_acc = correct / total
print("Test Acc: {:.4f}".format(test_acc))

#%% 정량적 판단 - train loss와 valid loss 비교

# 훈련 손실과 검증 손실 시각화
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, marker='o', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
