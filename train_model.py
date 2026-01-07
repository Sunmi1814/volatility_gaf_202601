# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import os

# 모듈 임포트
from src.data import FinancialDataset
from src.model import VolatilityCNN

def train():
    # --- Config ---
    CONFIG = {
            "window_size": 32,
            "future_days": 5,
            "img_size": 32,
            "batch_size": 64,
            "epochs": 15,
            "lr": 0.001,
            "tickers": ['SPY', 'QQQ', 'NVDA', 'AAPL', 'KO', 'JNJ', 'JPM', 'BAC', 'XOM', 'CVX']
        }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Training on: {DEVICE}")


    # 1. 데이터 로드
    dataset = FinancialDataset(
        tickers=CONFIG['tickers'],
        window_size=CONFIG['window_size'],
        future_days=CONFIG['future_days'],
        img_size=CONFIG['img_size']
    )
    
    # -------------------------------------------------------------------------
    # [수정 1] Split Logic & Description Update
    # -------------------------------------------------------------------------
    # 데이터셋 구조: [SPY 데이터 ... CVX 데이터] 순서로 연결되어 있음.
    # 따라서 단순히 인덱스로 8:2 분할을 하면, 앞쪽 종목(Tech/Index)으로 학습하고
    # 뒤쪽 종목(Energy/Finance 일부)으로 평가하게 됨.
    #
    # This split simulates a realistic deployment scenario:
    # - The model is trained on earlier samples (and specific sectors).
    # - It is validated on later samples and partially unseen asset structures.
    # - This provides a CONSERVATIVE estimate of generalization performance.
    # -------------------------------------------------------------------------
    dataset_size = len(dataset)
    split_idx = int(dataset_size * 0.8)
    
    indices = list(range(dataset_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    print(f" Train Samples: {len(train_idx)} | Val Samples: {len(val_idx)}")

    # -------------------------------------------------------------------------
    # [수정 2] Baseline Probability 사전 계산 (Data-Driven, Model-Independent)
    # 학습 루프 안에서 계산하면 epoch마다 값이 흔들릴 수 있음.
    # 데이터셋의 '본질적 위험도'를 미리 계산해두는 것이 통계적으로 올바름.
    # -------------------------------------------------------------------------
    print(" Calculating Baseline Risk from Validation Set...")
    # dataset[i][1]은 (1,) 형태의 텐서이므로 item()으로 값 추출
    val_targets = [dataset[i][1].item() for i in val_idx]
    baseline_prob = float(np.mean(val_targets))
    print(f"   -> Baseline Probability: {baseline_prob:.4f} (Validation Set Average)")
    
    # DataLoader 생성
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 2. 모델 준비
    model = VolatilityCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 3. 학습 루프
    best_auc = 0.0
    
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
            
        # Validation
        model.eval()
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                output = model(X)
                
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # Metrics
        binary_preds = (all_preds > 0.5).astype(int)
        val_acc = (binary_preds == all_targets).mean() * 100
        
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5
            
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Acc: {val_acc:.2f}% | "
              f"AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "artifacts/model.pt")
            print(f"   --> New Best Model Saved (AUC: {best_auc:.4f})")

    # 4. 결과 저장
    print(f" Final Best AUC: {best_auc:.4f}")
    
    # Metrics 저장 (사전 계산된 baseline 사용)
    metrics = {
        "best_auc": float(best_auc),
        "baseline_prob": baseline_prob  # 고정된 통계값 저장
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f)

    with open("artifacts/config.json", "w") as f:
        json.dump(CONFIG, f)
        
    print(" Training Complete. All artifacts saved.")

if __name__ == "__main__":
    train()
# %%
