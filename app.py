# app.py - 다중 상품 지원 버전 (상품 ID 입력 + 상품명 표시)
import streamlit as st
import numpy as np
import pandas as pd
import os
import gc
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 페이지 설정
# -----------------------------
# st.set_page_config(page_title="Instacart 상품 재구매 예측", layout="centered")
st.markdown("<h3 style='text-align: center; font-size: 24px;'>Instacart 상품 재구매 예측</h2>", unsafe_allow_html=True)
st.title("Instacart 상품 재구매 예측(LSTM)")

DATA_PATH = "data_sample"  # 필요시 수정

# -----------------------------
# 상품명 조회 함수
# -----------------------------
@st.cache_data
def get_product_name(product_id):
    products = pd.read_csv(f"{DATA_PATH}/products.csv")
    row = products[products['product_id'] == product_id]
    if not row.empty:
        return row['product_name'].iloc[0]
    return "알 수 없는 상품"

# -----------------------------
# 모델 파일명 생성
# -----------------------------
def get_model_file(product_id):
    return f"model_{product_id}.keras"

# -----------------------------
# 모델 학습 함수 (상품별)
# -----------------------------
def train_and_save_model(product_id):
    st.info(f"상품 {product_id} 모델을 처음 학습 중입니다... (30초 ~ 1분 소요)")
    progress = st.progress(0)
    
    # 데이터 로드
    orders = pd.read_csv(f"{DATA_PATH}/orders.csv",
                         usecols=['order_id', 'user_id', 'order_number', 'days_since_prior_order'])
    prior = pd.read_csv(f"{DATA_PATH}/order_products__prior.csv",
                        usecols=['order_id', 'product_id'])
    
    progress.progress(20)
    
    # 해당 상품 주문 추출
    target_orders = prior[prior['product_id'] == product_id]['order_id'].unique()
    if len(target_orders) == 0:
        st.error("이 상품에 대한 주문 기록이 없습니다.")
        st.stop()
    
    merged = orders.copy()
    merged['target'] = merged['order_id'].isin(target_orders).astype(int)
    merged['days_since_prior_order'] = merged['days_since_prior_order'].fillna(30).clip(upper=30)
    
    del orders, prior, target_orders
    gc.collect()
    
    progress.progress(40)
    
    # 빠른 시퀀스 생성 (최대 3000 샘플)
    seq_len = 5
    X, y = [], []
    
    all_users = merged['user_id'].unique()
    sample_users = random.sample(list(all_users), min(5000, len(all_users)))
    merged_sample = merged[merged['user_id'].isin(sample_users)]
    
    for user_id, user_df in merged_sample.groupby('user_id'):
        user_df = user_df.sort_values('order_number')
        targets = user_df['target'].values
        gaps = (user_df['days_since_prior_order'] / 30.0).clip(0, 1).values
        
        if len(targets) > seq_len:
            step = max(1, (len(targets) - seq_len) // 5)
            for i in range(0, len(targets) - seq_len, step):
                if len(X) >= 3000:
                    break
                X.append(np.stack([targets[i:i+seq_len], gaps[i:i+seq_len]], axis=1))
                y.append(targets[i+seq_len])
        
        if len(X) >= 3000:
            break
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) < 50:
        st.warning("학습 데이터가 부족합니다. 예측 정확도가 낮을 수 있습니다.")
    
    progress.progress(80)
    
    # 가벼운 모델
    model = Sequential([
        LSTM(16, input_shape=(5, 2)),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X, y, epochs=30, batch_size=256, validation_split=0.2,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=0)
    
    # 저장
    model_file = get_model_file(product_id)
    model.save(model_file)
    progress.progress(100)
    st.success(f"✅ {product_id} 모델 학습 및 저장 완료!")

# -----------------------------
# 모델 로드 (상품별)
# -----------------------------
@st.cache_resource
def get_model(product_id):
    model_file = get_model_file(product_id)
    if os.path.exists(model_file):
        return load_model(model_file)
    else:
        train_and_save_model(product_id)
        return load_model(model_file)

# -----------------------------
# UI: 상품 ID 입력 + 상품명 표시
# -----------------------------
st.markdown("### 예측할 상품 ID 입력")
st.caption("추천ID: 13176, 21137, 24852,21903,47209 등 인기 상품 사용 권장")
product_id = st.number_input("상품 ID (product_id)", min_value=1, value=24852, step=1)

# 상품명 조회 및 표시
with st.spinner("상품명 조회 중..."):
    product_name = get_product_name(product_id)

st.success(f"**선택 상품:** {product_id} ({product_name})")

model = get_model(product_id)

st.markdown(f"<h3 style='text-align: center;'>{product_name} 재구매 확률 예측</h3>", unsafe_allow_html=True)

st.info("최근 5회 주문 내역을 입력하세요 (오래된 순 → 최근 순)")

cols = st.columns(5)
purchase_history = []
gap_history = []

for i, col in enumerate(cols):
    with col:
        st.markdown(f"**주문 {i+1}**")
        purchase = st.radio("구매", [0, 1], index=0, horizontal=True, key=f"p{i}_{product_id}")
        gap = st.selectbox("간격(일)", [1, 3, 7, 14, 21, 30], index=5, key=f"g{i}_{product_id}")
        purchase_history.append(purchase)
        gap_history.append(gap / 30.0)

if st.button("🚀 예측하기", type="primary", use_container_width=True):
    input_seq = np.stack([purchase_history, gap_history], axis=1).reshape(1, 5, 2)
    
    prob = float(model.predict(input_seq, verbose=0)[0][0])
    prob_percent = prob * 100

    st.markdown(f"""
    <h2 style='text-align: center; color: #1976D2;'>
        다음 주문에서 <b>{product_name}</b> 구매 확률<br>
        <b style='font-size: 2em;'>{prob_percent:.1f}%</b>
    </h2>
    """, unsafe_allow_html=True)

    if prob_percent >= 70:
        st.success("🟢 매우 높음 → 쿠폰 발송 강력 추천!")
    elif prob_percent >= 50:
        st.warning("🟡 높음 → 추천 상품 노출")
    elif prob_percent >= 30:
        st.info("🔵 보통 → 일반 추천")
    else:
        st.error("🔴 낮음 → 다른 상품 고려")

    with st.expander("입력 내역 확인"):
        gap_options = [1, 3, 7, 14, 21, 30]
        original_gaps = [min(gap_options, key=lambda x: abs(x - round(g * 30))) for g in gap_history]
        
        df = pd.DataFrame({
            "주문": [f"주문 {i+1} (오래된→최근)" for i in range(5)],
            "구매": ["구매" if p else "미구매" for p in purchase_history],
            "간격(일)": original_gaps
        })
        st.table(df)

st.caption("※ 상품별 전용 LSTM 모델 사용 | 최초 예측 시 1분 내 학습 후 저장")
st.caption("※ 인기 상품일수록 예측 정확도 높음 (예: 13176 - Bag of Organic Bananas, 21137 - Organic Strawberries 등)")