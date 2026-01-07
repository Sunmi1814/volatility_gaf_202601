# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import json
import os

from src.gaf import GAFEncoder
from src.model import VolatilityCNN

# 페이지 설정
st.set_page_config(page_title="Market Regime Detector", layout="wide")

st.title("📊 Market Volatility Regime Detector")
st.markdown("### AI-Powered Crisis Early Warning System")

# 1. 설정 파일 로드 (학습된 설정과 동기화)
CONFIG_PATH = "artifacts/config.json"
MODEL_PATH = "artifacts/model.pt"

if not os.path.exists(CONFIG_PATH) or not os.path.exists(MODEL_PATH):
    st.error("❌ Model or Config not found. Please run 'train_model.py' first.")
    st.stop()

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# 학습 때 사용한 Window Size 가져오기 (32)
WINDOW_SIZE = config.get("window_size", 32)
IMG_SIZE = config.get("img_size", 32)

# 사이드바
st.sidebar.header("Control Panel")
ticker = st.sidebar.text_input("Ticker Symbol", "NVDA").upper()
st.sidebar.info(f"Model Window Size: {WINDOW_SIZE} days")

if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # 데이터 로드 (넉넉하게 6개월치)
            df = yf.download(ticker, period="6mo", progress=False)
            
            # yfinance 호환성 처리
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    close = df['Close'][ticker]
                except KeyError:
                    close = df['Close'].iloc[:, 0]
            else:
                close = df['Close']
            
            # 로그 수익률 계산
            log_ret = np.log(close / close.shift(1)).dropna()

            if len(log_ret) < WINDOW_SIZE:
                st.error(f"Not enough data. Need at least {WINDOW_SIZE} trading days.")
            else:
                # 2. 가장 최근 데이터 인코딩
                # GAFEncoder는 내부적으로 Fixed Scaling (+/- 5%)을 수행함
                recent_data = log_ret[-WINDOW_SIZE:].values
                
                encoder = GAFEncoder(IMG_SIZE)
                # GAF 변환 (1D -> 2D Image)
                # 모델은 (Price, Volume) 2채널을 학습했으나, 데모에서는 편의상 Price만 시각화하거나
                # 혹은 모델 입력에 맞춰 Volume도 가져와야 함.
                # -> 여기서는 모델 입력 형식을 맞추기 위해 Volume도 가져옵니다.
                
                # Volume 데이터 로드
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        volume = df['Volume'][ticker]
                    except KeyError:
                        volume = df['Volume'].iloc[:, 0]
                else:
                    volume = df['Volume']
                
                # Volume 전처리
                log_vol = np.log(volume + 1e-8).dropna()
                recent_vol = log_vol[-WINDOW_SIZE:].values
                
                # Volume Scaling (MinMax)
                v_min, v_max = recent_vol.min(), recent_vol.max()
                if v_max - v_min == 0:
                    vol_scaled = np.zeros_like(recent_vol)
                else:
                    vol_scaled = 2 * (recent_vol - v_min) / (v_max - v_min) - 1
                
                # 인코딩
                gaf_ret = encoder.encode(recent_data) # Price Channel
                gaf_vol = encoder.encode(vol_scaled)  # Volume Channel
                
                # 입력 텐서 만들기 (1, 2, 32, 32)
                img_stack = np.stack([gaf_ret, gaf_vol], axis=0)
                x_tensor = torch.tensor(img_stack).float().unsqueeze(0)

                # 3. 모델 로드 및 추론
                model = VolatilityCNN()
                # CNN 구조가 학습 때와 동일해야 함 (model.py 수정 없으므로 OK)
                model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
                model.eval()
                
                with torch.no_grad():
                    logits = model(x_tensor)
                    prob = torch.sigmoid(logits).item()
                
                # 4. 결과 시각화
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Market Structure (GAF)")
                    st.caption("Left: Price Structure | Right: Volume Structure")
                    
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    
                    # Price GAF
                    axs[0].imshow(gaf_ret, cmap='rainbow', vmin=-1, vmax=1, origin='lower')
                    axs[0].set_title("Price Dynamics")
                    axs[0].axis('off')
                    
                    # Volume GAF
                    axs[1].imshow(gaf_vol, cmap='viridis', vmin=-1, vmax=1, origin='lower')
                    axs[1].set_title("Volume Dynamics")
                    axs[1].axis('off')
                    
                    st.pyplot(fig)
                    
                with col2:
                    st.subheader("AI Risk Assessment")
                    
                    gauge_color = "green"
                    if prob > 0.7: gauge_color = "red"
                    elif prob > 0.4: gauge_color = "orange"
                    
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <h1 style="color: {gauge_color}; font-size: 60px;">{prob*100:.1f}%</h1>
                        <p>High Volatility Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if prob > 0.7:
                        st.error("🚨 **CRITICAL WARNING**\n\n시장 구조가 불안정합니다. 변동성 폭발 가능성이 높습니다.")
                    elif prob > 0.4:
                        st.warning("⚠️ **CAUTION**\n\n변동성 확대 조짐이 보입니다. 모니터링이 필요합니다.")
                    else:
                        st.success("✅ **STABLE**\n\n시장 구조가 안정적입니다.")
                
                st.divider()
                st.subheader("Recent Price Trend")
                st.line_chart(close[-60:])
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("👈 Enter a ticker symbol (e.g., NVDA, BTC-USD) and click 'Run Analysis'")