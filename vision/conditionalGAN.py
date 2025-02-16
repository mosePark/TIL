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

#%%
# [1] KeyWatchDataset: 폴더 내의 GIF 이미지 읽기, 파일명에 따라 라벨 지정
class KeyWatchDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = glob.glob(os.path.join(folder, "*.gif"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # 이미지를 흑백으로 로드
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        # 파일명에 따라 라벨 지정: key -> 0, watch -> 1
        filename = os.path.basename(path).lower()
        if "key" in filename:
            label = 0
        elif "watch" in filename:
            label = 1
        else:
            raise ValueError("Unknown label in file: " + filename)
        return img, label

#%%
# [2] Conditional GAN 모델 정의
# (A) cGAN Generator: latent vector와 레이블 정보를 입력받아 이미지를 생성  
class cGANGenerator(nn.Module):
    def __init__(self, num_classes=2, latent_dim=100, feature_g=64):
        super(cGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # 입력 채널 = latent_dim + num_classes (레이블은 원-핫으로 표현)
        self.main = nn.Sequential(
            # (latent_dim+num_classes) x 1 x 1 → feature_g*8 x 4 x 4
            nn.ConvTranspose2d(latent_dim + num_classes, feature_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),
            # feature_g*8 x 4 x 4 → feature_g*4 x 8 x 8
            nn.ConvTranspose2d(feature_g*8, feature_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),
            # feature_g*4 x 8 x 8 → feature_g*2 x 16 x 16
            nn.ConvTranspose2d(feature_g*4, feature_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),
            # feature_g*2 x 16 x 16 → feature_g x 32 x 32
            nn.ConvTranspose2d(feature_g*2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # feature_g x 32 x 32 → 1 x 64 x 64 (흑백 이미지)
            nn.ConvTranspose2d(feature_g, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z, labels):
        # z: (N, latent_dim, 1, 1)
        # labels: (N,) 정수형, 0 또는 1
        N = z.size(0)
        # 레이블을 원-핫 벡터 (N, num_classes, 1, 1)로 변환
        one_hot = torch.zeros(N, self.num_classes, 1, 1, device=z.device)
        one_hot[range(N), labels] = 1.0
        # 채널 차원에서 concat: 결과 (N, latent_dim+num_classes, 1, 1)
        x = torch.cat([z, one_hot], dim=1)
        return self.main(x)

# (B) cGAN Discriminator: 이미지와 레이블 정보를 입력받아 진짜/가짜 판별  
class cGANDiscriminator(nn.Module):
    def __init__(self, num_classes=2, feature_d=64):
        super(cGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        # 입력 채널 = 1 (이미지) + num_classes (레이블의 원-핫 맵)
        self.main = nn.Sequential(
            # (1+num_classes) x 64 x 64 → feature_d x 32 x 32
            nn.Conv2d(1 + num_classes, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_d x 32 x 32 → feature_d*2 x 16 x 16
            nn.Conv2d(feature_d, feature_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*2),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_d*2 x 16 x 16 → feature_d*4 x 8 x 8
            nn.Conv2d(feature_d*2, feature_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*4),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_d*4 x 8 x 8 → feature_d*8 x 4 x 4
            nn.Conv2d(feature_d*4, feature_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*8),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_d*8 x 4 x 4 → 1 x 1 x 1
            nn.Conv2d(feature_d*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        N = img.size(0)
        # 레이블을 64x64 크기의 원-핫 맵으로 확장: (N, num_classes, 64, 64)
        one_hot = torch.zeros(N, self.num_classes, img.size(2), img.size(3), device=img.device)
        for i in range(N):
            one_hot[i, labels[i], :, :] = 1.0
        # 이미지와 레이블 맵을 채널 차원에서 concat: (N, 1+num_classes, 64, 64)
        x = torch.cat([img, one_hot], dim=1)
        return self.main(x)

#%%
# [3] 하이퍼파라미터 및 데이터셋/로더 설정
# 이미지 크기, latent_dim, 배치 크기 등
image_size = 64
latent_dim = 100
lr = 0.0002
beta1 = 0.5
num_epochs = 5   # 예시: 에포크 수 (데이터와 상황에 따라 조정)
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로: train_aug 폴더 (key와 watch 이미지가 섞여 있음)
data_folder = ".../train_aug"

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 흑백이므로 평균/표준편차가 (0.5,)
])

dataset = KeyWatchDataset(data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset size: {len(dataset)}")

#%%
# [4] 모델 생성 및 초기화 (Conditional GAN)
netG = cGANGenerator(num_classes=2, latent_dim=latent_dim, feature_g=64).to(device)
netD = cGANDiscriminator(num_classes=2, feature_d=64).to(device)

# 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

#%%
# [5] Loss 함수와 옵티마이저
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

#%%
# [6] 학습 루프
# 고정 노이즈와 고정 레이블 (시각화용): 예를 들어 8개는 key(0), 8개는 watch(1)
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
fixed_labels = torch.tensor([0]*8 + [1]*8, device=device)

for epoch in range(num_epochs):
    t0 = time.time()
    for i, (real_imgs, real_labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)
        b_size = real_imgs.size(0)
        
        # [A] 판별자 학습
        netD.zero_grad()
        # real 이미지에 대한 손실
        label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
        output_real = netD(real_imgs, real_labels).view(-1)
        lossD_real = criterion(output_real, label_real)
        lossD_real.backward()
        
        # fake 이미지 생성 (랜덤 noise와 랜덤 label)
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_labels = torch.randint(0, 2, (b_size,), device=device)  # 0 또는 1 랜덤 샘플
        fake_imgs = netG(noise, fake_labels)
        label_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
        output_fake = netD(fake_imgs.detach(), fake_labels).view(-1)
        lossD_fake = criterion(output_fake, label_fake)
        lossD_fake.backward()
        
        lossD = lossD_real + lossD_fake
        optimizerD.step()
        
        # [B] 생성자 학습
        netG.zero_grad()
        # 생성자가 만든 fake 이미지를 real로 인식하도록 유도
        output_fake_for_G = netD(fake_imgs, fake_labels).view(-1)
        label_g = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
        lossG = criterion(output_fake_for_G, label_g)
        lossG.backward()
        optimizerG.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(dataloader)}] "
                  f"LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")
    
    t1 = time.time()
    print(f"--> Epoch {epoch+1} finished in {t1 - t0:.2f} sec")
    
    # 중간에 고정 noise와 고정 label을 사용해 이미지 생성 및 시각화
    netG.eval()
    with torch.no_grad():
        fake_samples = netG(fixed_noise, fixed_labels).cpu()
    netG.train()
    
    # 정규화 해제: [-1,1] -> [0,1]
    fake_samples = (fake_samples * 0.5) + 0.5
    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    for ax, img in zip(axes.flatten(), fake_samples):
        ax.imshow(img.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.suptitle(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.show()

print("GAN 학습 완")
    denoised_img.save(filename)

print(f"{num_to_generate} synthetic images (denoised) saved to '{save_folder}'.")
# %%
