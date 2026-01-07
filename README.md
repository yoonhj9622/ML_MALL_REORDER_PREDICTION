# ML_MALL_REORDER_PREDICTION

이 프로젝트는 쇼핑몰 고객의 재구매 여부를 예측하기 위한 머신러닝 프로젝트입니다. 고객의 과거 구매 이력 및 행동 데이터를 분석하여 특정 상품을 다시 구매할지 여부를 이진 분류(Binary Classification)하는 모델을 구축하는 것을 목표로 합니다.
브라우저에서 확인 : https://mlmallreorderprediction-zcbblyk3krgalkyueov6sr.streamlit.app/
## 1. 프로젝트 개요

* **목표**: 고객 데이터 및 주문 데이터를 활용한 재구매(Reorder) 예측
* **주요 작업**:
* 데이터 전처리 및 EDA (탐색적 데이터 분석)
* 피처 엔지니어링 (Feature Engineering)
* 머신러닝 모델 학습 및 성능 평가



## 2. 데이터셋 구성

프로젝트에서 사용된 주요 데이터 컬럼은 다음과 같습니다:

* `order_id`: 주문 고유 번호
* `user_id`: 사용자 고유 번호
* `product_id`: 상품 고유 번호
* `reordered`: 재구매 여부 (Target: 0 또는 1)
* 기타 주문 시간, 요일, 이전 주문과의 간격 등

## 3. 기술 스택

* **Language**: Python
* **Libraries**:
* `Pandas`, `NumPy`: 데이터 핸들링
* `Matplotlib`, `Seaborn`: 데이터 시각화
* `Scikit-learn`: 머신러닝 모델 구축 및 평가
* `XGBoost`, `LightGBM`: 고성능 분류 모델 사용



## 4. 분석 프로세스

1. **데이터 로드 및 결합**: 여러 개로 나뉘어진 데이터 테이블(orders, products, aisles 등)을 통합.
2. **EDA**:
* 시간대별/요일별 주문량 분석
* 가장 많이 재구매되는 상품군 파악


3. **피처 엔지니어링**:
* 사용자별 평균 주문 간격
* 상품별 재구매율
* 사용자-상품 간의 친밀도(구매 빈도) 계산


4. **모델링**: XGBoost 또는 LightGBM 모델을 활용하여 학습 진행.
5. **평가**: F1-Score 및 AUC-ROC 곡선을 통한 모델 성능 검증.

## 5. 결과 및 시사점

* 사용자의 과거 구매 패턴이 재구매 예측에 가장 중요한 요소임을 확인.
* 특정 주기(예: 7일, 30일)마다 반복되는 구매 성향이 모델 성능 향상에 기여함.

---

### 실행 방법

```bash
git clone https://github.com/yoonhj9622/ML_MALL_REORDER_PREDICTION.git

```
