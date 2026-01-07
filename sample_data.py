# sample_data.py - 데이터 샘플링해서 작게 저장
import pandas as pd

DATA_PATH = "../data"  # 원본 경로
SAMPLE_PATH = "data_sample"  # 새 폴더 생성

# 폴더 생성
import os
if not os.path.exists(SAMPLE_PATH):
    os.makedirs(SAMPLE_PATH)

# 1. orders.csv - 10% 샘플
orders = pd.read_csv(f"{DATA_PATH}/orders.csv")
orders_sample = orders.sample(frac=0.1, random_state=42)
orders_sample.to_csv(f"{SAMPLE_PATH}/orders.csv", index=False)
print(f"orders: {orders.shape} → {orders_sample.shape}")

# 2. order_products__prior.csv - 1% 샘플 (가장 큼!)
prior = pd.read_csv(f"{DATA_PATH}/order_products__prior.csv")
prior_sample = prior.sample(frac=0.01, random_state=42)
prior_sample.to_csv(f"{SAMPLE_PATH}/order_products__prior.csv", index=False)
print(f"prior: {prior.shape} → {prior_sample.shape}")

# 3. products.csv - 전체 (작아서 괜찮음)
products = pd.read_csv(f"{DATA_PATH}/products.csv")
products.to_csv(f"{SAMPLE_PATH}/products.csv", index=False)
print(f"products: {products.shape}")

print("샘플 데이터 저장 완료! data_sample 폴더를 GitHub에 올리세요.")