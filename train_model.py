# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score # 🚀 핵심 추가
import json
import os

# 모듈 임포트
from src.data import FinancialDataset
from src.model import VolatilityCNN

def train():
    # --- Config ---
    CONFIG = {
            "window_size": 32,  # 👈 기존 20에서 32로 수정 (img_size와 맞춤)
            "future_days": 5,
            "img_size": 32,     # 얘랑 크기가 같거나 커야 함
            "batch_size": 64,
            "epochs": 15,
            "lr": 0.001,
            "tickers": ['SPY', 'QQQ', 'NVDA', 'AAPL', 'KO', 'JNJ', 'JPM', 'BAC', 'XOM', 'CVX']
        }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {DEVICE}")

    # 1. 데이터 로드
    dataset = FinancialDataset(
        tickers=CONFIG['tickers'],
        window_size=CONFIG['window_size'],
        future_days=CONFIG['future_days'],
        img_size=CONFIG['img_size']
    )
    
    # 🚨 [수정 1] Time-Aware Split (Data Leakage 방지)
    # 셔플을 끄고, 데이터의 뒤쪽 20%를 검증셋으로 사용합니다.
    # 우리 데이터셋은 종목별로 정렬되어 있으므로, 이 방식은 "본 적 없는 종목(Unseen Tickers)"에 대한
    # 일반화 성능을 테스트하는 'Cross-Sectional Split' 효과도 가집니다. (매우 강력한 검증)
    dataset_size = len(dataset)
    split_idx = int(dataset_size * 0.8)
    
    # 인덱스 리스트 생성 (순서 유지)
    indices = list(range(dataset_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    print(f"📊 Train Samples: {len(train_idx)} | Val Samples: {len(val_idx)}")
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=CONFIG['batch_size'], shuffle=True)
    # 검증 때는 섞지 않음
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 2. 모델 준비
    model = VolatilityCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 3. 학습 루프
    best_auc = 0.0 # Accuracy 대신 AUC를 기준으로 모델 저장
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 🚨 [수정 2] 정밀 검증 (ROC-AUC 추가)
        model.eval()
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                output = model(X)
                
                # AUC 계산을 위해 확률값과 정답을 모음
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # 리스트 -> 넘파이 변환
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # Metric 계산
        # 1. Accuracy (0.5 기준)
        binary_preds = (all_preds > 0.5).astype(int)
        val_acc = (binary_preds == all_targets).mean() * 100
        
        # 2. ROC-AUC (변별력 핵심 지표)
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5 # 데이터가 한 클래스만 있을 경우 예외 처리
            
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Acc: {val_acc:.2f}% | "
              f"AUC: {val_auc:.4f}") # AUC 출력
        
        # AUC가 가장 높을 때 모델 저장 (더 신뢰할 수 있는 기준)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "artifacts/model.pt")
            print(f"   --> New Best Model Saved (AUC: {best_auc:.4f})")

    # 4. 결과 저장
    print(f"🏆 Final Best AUC: {best_auc:.4f}")
    
    with open("artifacts/config.json", "w") as f:
        json.dump(CONFIG, f)
        
    print("✅ Training Complete.")

if __name__ == "__main__":
    train()
# %%
