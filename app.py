import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Churn UAS", layout="wide")
st.title("📊 Prediksi Churn Pelanggan UAS")

# 1. Load Model + Scaler
model = joblib.load('model_churn_terbaik.pkl')
scaler = joblib.load('scaler.pkl')

# 2. WAJIB ISI INI DENGAN KOLOM KAMU
# Jalankan: print(X.columns.tolist()) di cell atas, lalu copy paste hasilnya ke sini
FEATURE_COLUMNS = [
    'age', 'total_visits', 'avg_session_time', 'pages_per_session',
    #...lanjutkan semua kolom dari X.columns kamu sampai habis
]

st.write("Isi semua fitur di bawah ini. Kategorikal: 0 atau 1.")

# 3. Input Form
input_data = {}
cols = st.columns(5)
for i, col in enumerate(FEATURE_COLUMNS):
    with cols[i % 5]:
        input_data[col] = st.number_input(col, value=0.0, format="%.4f", key=col)

if st.button("🔮 Prediksi Sekarang"):
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("🚨 Hasil: PELANGGAN AKAN CHURN = 1")
    else:
        st.success("✅ Hasil: PELANGGAN TIDAK CHURN = 0")
