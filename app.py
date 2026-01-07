# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import json
import os
import random

from src.gaf import GAFEncoder
from src.model import VolatilityCNN

# ---------------------------------------------------------
# [1] 재현성 확보 (Reproducibility)
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 페이지 설정
st.set_page_config(page_title="Market Regime Detector", layout="wide")

st.title("📊 Market Volatility Regime Detector")
st.markdown("### AI-Powered Crisis Early Warning System")

# 1. 설정 및 파일 로드
CONFIG_PATH = "artifacts/config.json"
MODEL_PATH = "artifacts/model.pt"
METRICS_PATH = "artifacts/metrics.json"

# 파일 체크
if not os.path.exists(CONFIG_PATH) or not os.path.exists(MODEL_PATH):
    st.error(" Model or Config not found. Please run 'train_model.py' first.")
    st.stop()

if not os.path.exists(METRICS_PATH): 
    st.error(" Metrics not found. Please run 'train_model.py' again.")
    st.stop()

# 파일 읽기
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# Baseline & Metric 로드
baseline_prob = metrics.get("baseline_prob", 0.25)
best_auc = metrics.get("best_auc", 0.0)

WINDOW_SIZE = config.get("window_size", 32)
IMG_SIZE = config.get("img_size", 32)

# ---------------------------------------------------------
# [2] 모델 로드 캐싱 (Performance Optimization)
# 발표 멘트: "매번 모델을 로드하면 느려지므로, 캐싱을 통해 추론 속도를 최적화했습니다."
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = VolatilityCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# 사이드바 설정
st.sidebar.header("Control Panel")
ticker = st.sidebar.text_input("Ticker Symbol", "NVDA").upper()

st.sidebar.divider()
st.sidebar.header("Model Info")
st.sidebar.info(f"Window Size: {WINDOW_SIZE} days")
st.sidebar.markdown(f"""
**Model AUC:** `{best_auc:.4f}`
**Baseline Risk:** `{baseline_prob*100:.1f}%`
*(Historical Avg. Probability)*
""")

if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # 데이터 로드
            df = yf.download(
                ticker,
                period="6mo",
                auto_adjust=True,
                progress=False
            )

            df = df.dropna()
            df = df.iloc[:-1]  # 마지막 하루 제거 (미완성 캔들 제거)
            
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    close = df['Close'][ticker]
                except KeyError:
                    close = df['Close'].iloc[:, 0]
            else:
                close = df['Close']
            
            # 로그 수익률
            log_ret = np.log(close / close.shift(1)).dropna()

            if len(log_ret) < WINDOW_SIZE:
                st.error(f"Not enough data. Need at least {WINDOW_SIZE} trading days.")
            else:
                # 2. 데이터 전처리
                recent_data = log_ret[-WINDOW_SIZE:].values

                # [Fixed Scaling] +/- 5% 기준
                recent_data_scaled = np.clip(recent_data / 0.05, -1, 1)

                # Volume 처리
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        volume = df['Volume'][ticker]
                    except KeyError:
                        volume = df['Volume'].iloc[:, 0]
                else:
                    volume = df['Volume']
                
                log_vol = np.log(volume + 1e-8).dropna()
                recent_vol = log_vol[-WINDOW_SIZE:].values
                
                v_min, v_max = recent_vol.min(), recent_vol.max()
                if v_max - v_min == 0:
                    vol_scaled = np.zeros_like(recent_vol)
                else:
                    vol_scaled = 2 * (recent_vol - v_min) / (v_max - v_min) - 1
                
                # 인코딩
                encoder = GAFEncoder(IMG_SIZE)
                gaf_ret = encoder.encode(recent_data_scaled)
                gaf_vol = encoder.encode(vol_scaled)
                
                # 텐서 변환
                img_stack = np.stack([gaf_ret, gaf_vol], axis=0)
                x_tensor = torch.tensor(img_stack).float().unsqueeze(0)

                # 3. 모델 추론 (캐싱된 모델 사용)
                model = load_model()
                
                with torch.no_grad():
                    logits = model(x_tensor)
                    prob = torch.sigmoid(logits).item()
                
                # 4. 결과 시각화
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Market Structure (GAF)")
                    st.caption("Left: Price Structure | Right: Volume Structure")
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    axs[0].imshow(gaf_ret, cmap='rainbow', vmin=-1, vmax=1, origin='lower')
                    axs[0].axis('off'); axs[0].set_title("Price")
                    axs[1].imshow(gaf_vol, cmap='viridis', vmin=-1, vmax=1, origin='lower')
                    axs[1].axis('off'); axs[1].set_title("Volume")
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

                    # ---------------------------------------------------------
                    # [3] 통계적 해석 강화 & 안전한 나눗셈 (Numerical Stability)
                    # ---------------------------------------------------------
                    eps = 1e-6 # 0 나누기 방지용 엡실론
                    ratio = prob / max(baseline_prob, eps)
                    delta = prob - baseline_prob
                    
                    # 수치 해석 박스
                    st.info(f"""
                    **Statistical Insight:**
                    - Baseline Ratio: **{ratio:.2f}x** (vs Avg)
                    - Deviation: **{delta*100:+.1f}pp** (percentage points)
                    """)
                    
                    if prob > 0.7:
                        st.error("🚨 **CRITICAL** : 시장 구조가 매우 불안정합니다.")
                    elif prob > 0.4:
                        st.warning("⚠️ **CAUTION** : 변동성 확대 조짐이 보입니다.")
                    else:
                        st.success("✅ **STABLE** : 안정적인 시장 구조입니다.")
                
                st.divider()
                st.subheader("Recent Price Trend")
                st.line_chart(close[-60:])
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Enter a ticker symbol and click 'Run Analysis'")