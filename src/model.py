import torch
import torch.nn as nn

class VolatilityCNN(nn.Module):
    def __init__(self):
        super(VolatilityCNN, self).__init__()
        
        # Input: (Batch, 2, 32, 32) -> Channel 0: Price, Channel 1: Volume
        
        # Layer 1: 특징 추출 (Low-level features)
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2) # 32 -> 16
        )
        
        # Layer 2: 특징 결합 (Mid-level features)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 16 -> 8
        )
        
        # Layer 3: 고차원 패턴 인식 (High-level features)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # 8 -> 4
        )
        
        # Fully Connected: 분류 (0 or 1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 과적합 방지
            nn.Linear(128, 1),
            nn.Sigmoid() # 확률값 출력 (0~1)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)
        return out