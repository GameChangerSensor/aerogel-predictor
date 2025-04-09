import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ✅ 모델 및 스케일러 로드 (compile=False로 에러 방지)
model = load_model("saved_model", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ✅ 입력값 허용 범위 (학습 기반)
FREQ_MIN, FREQ_MAX = 10, 100000        # Hz
IMP_MIN, IMP_MAX = 1000, 50000         # Ω
TIME_MIN, TIME_MAX = 0, 1440           # 분

# ✅ Streamlit UI
st.title("💠 에어로겔 물성 예측기")
st.markdown("Pd 센서 데이터를 기반으로 **표면적 (m²/g)**, **기공 직경 (nm)**, **기공 부피 (cm³/g)**를 예측합니다.")
st.caption("🛈 입력 단위: Frequency (Hz), Impedance (Ω), Time (minutes)")

# ✅ 사용자 입력
freq = st.number_input("Frequency (Hz)", min_value=0.0, value=100.0)
impedance = st.number_input("Impedance (Ω)", min_value=0.0, value=2000.0)
time = st.number_input("Time (minutes)", min_value=0.0, value=10.0)

# ✅ 예측 버튼 클릭 시 유효성 검사 + 예측 수행
if st.button("Predict"):
    if not (FREQ_MIN <= freq <= FREQ_MAX):
        st.error(f"⚠️ Frequency 값은 {FREQ_MIN} ~ {FREQ_MAX} Hz 사이여야 합니다. (입력: {freq})")
    elif not (IMP_MIN <= impedance <= IMP_MAX):
        st.error(f"⚠️ Impedance 값은 {IMP_MIN} ~ {IMP_MAX} Ω 사이여야 합니다. (입력: {impedance})")
    elif not (TIME_MIN <= time <= TIME_MAX):
        st.error(f"⚠️ Time 값은 {TIME_MIN} ~ {TIME_MAX} 분 사이여야 합니다. (입력: {time})")
    else:
        # 🔹 입력값 전처리 및 예측
        input_data = np.array([[freq, impedance, time]])
        input_scaled = scaler_X.transform(input_data)
        pred_scaled = model.predict(input_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)

        # 🔹 예측 결과 출력
        st.subheader("📈 예측 결과")
        st.write(f"**Surface Area**: {pred[0][0]:.2f} m²/g")
        st.write(f"**Pore Diameter**: {pred[0][1]:.2f} nm")
        st.write(f"**Pore Volume**: {pred[0][2]:.4f} cm³/g")


