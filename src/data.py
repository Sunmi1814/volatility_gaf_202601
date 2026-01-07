# src/data.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import yfinance as yf
from .gaf import GAFEncoder  # 같은 폴더 내 gaf.py 참조

class FinancialDataset(Dataset):
    def __init__(self, tickers=None, start_date='2018-01-01', window_size=20, future_days=5, img_size=32):
        if tickers is None:
            # Sector-Balanced Ticker List (10 종목)
            tickers = [
                'SPY', 'QQQ',       # Index
                'NVDA', 'AAPL',     # Tech
                'KO', 'JNJ',        # Defensive
                'JPM', 'BAC',       # Finance
                'XOM', 'CVX'        # Energy
            ]
            
        self.window_size = window_size
        self.encoder = GAFEncoder(img_size)
        
        self.X = []
        self.y = []
        
        print(f"Build Dataset: {start_date} ~ Now | Window: {window_size}d")
        
        for t in tickers:
            self._process_ticker(t, start_date, window_size, future_days)
            
        if len(self.X) > 0:
            self.X = np.concatenate(self.X, axis=0)
            self.y = np.concatenate(self.y, axis=0)
            print(f"✅ Total Data Loaded: {len(self.X)} samples")
        else:
            print("⚠️ No data loaded. Please check your internet connection or ticker symbols.")

    def _process_ticker(self, ticker, start, window, future):
        try:
            # 1. 데이터 다운로드 (yfinance 최신 버그 방지 로직 적용)
            data = yf.download(ticker, start=start, progress=False)
            
            if len(data) < window + future: return

            # Multi-index 처리 (yfinance가 가끔 (Price, Ticker) 형태로 줌)
            if isinstance(data.columns, pd.MultiIndex):
                # 해당 티커의 컬럼만 추출 시도
                try:
                    close = data['Close'][ticker]
                    volume = data['Volume'][ticker]
                except KeyError:
                    # 티커 이름 없이 레벨만 있는 경우 (단일 종목 다운로드 시)
                    close = data['Close'].iloc[:, 0]
                    volume = data['Volume'].iloc[:, 0]
            else:
                close = data['Close']
                volume = data['Volume']

            # 2. Log Returns & Log Volume (정규화 기초)
            # 0 나누기 방지를 위해 1e-8 추가
            log_ret = np.log(close / close.shift(1) + 1e-8)
            log_vol = np.log(volume + 1e-8)
            
            df = pd.DataFrame({'Ret': log_ret, 'Vol': log_vol}).dropna()
            
            # 3. Labeling (각 종목의 과거 변동성 상위 25% = 위험)
            future_vol = df['Ret'].rolling(window=future).std().shift(-future)
            threshold = future_vol.quantile(0.75)
            
            # 4. Scaling Strategy (핵심: Fixed Scaling)
            # Returns: +/- 5%를 넘어가면 -1, 1로 고정 (시장 충격 보존)
            ret_scaled = np.clip(df['Ret'].values / 0.05, -1, 1)
            
            # Volume: 종목별 MinMax (상대적 변화량이므로 MinMax OK)
            v_min, v_max = df['Vol'].min(), df['Vol'].max()
            vol_scaled = 2 * (df['Vol'].values - v_min) / (v_max - v_min) - 1
            
            # 5. Generate Samples
            ticker_imgs = []
            ticker_labels = []
            
            vals_vol = future_vol.values
            
            for i in range(window, len(df) - future):
                target = vals_vol[i]
                if np.isnan(target): continue
                
                label = 1 if target >= threshold else 0
                
                # GAF Encoding (주의: transform이 아니라 encode 메서드 사용)
                gaf_ret = self.encoder.encode(ret_scaled[i-window:i])
                gaf_vol = self.encoder.encode(vol_scaled[i-window:i])
                
                # Stack (Channel 2, Height 32, Width 32)
                img = np.stack([gaf_ret, gaf_vol], axis=0)
                
                ticker_imgs.append(img)
                ticker_labels.append(label)
            
            if len(ticker_imgs) > 0:
                self.X.append(np.array(ticker_imgs))
                self.y.append(np.array(ticker_labels))
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)