import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================
# 1. Konfigurasi & CSS
# ============================
st.set_page_config(page_title="Prediksi Churn Pelanggan", page_icon="🔮", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"]{ background-color:#EEF2F7; }
.stButton>button{ background-color:#4F46E5; color:white; border-radius:8px; width:100%; height:45px; font-size:16px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ============================
# 2. Load Model, Scaler, FEATURES, BASELINE
# ============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_churn_terbaik.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl") # WAJIB: hasil dari notebook EDA
    baseline_df = pd.read_csv("baseline_churn_input.csv") # WAJIB: 1 baris median
    return model, scaler, feature_columns, baseline_df

model, scaler, FEATURE_COLUMNS, baseline_df = load_artifacts()

# ============================
# 3. SIDEBAR INPUT - Fokus ke fitur yang ngaruh ke churn
# ============================
st.sidebar.title("Input Fitur Pelanggan")
st.sidebar.caption("Ubah nilai di bawah untuk melihat perubahan prediksi")

age = st.sidebar.number_input("Usia", 18, 80, int(baseline_df['age'].iloc[0]))
premium = st.sidebar.selectbox("Premium User", ["Tidak","Ya"], index=int(baseline_df['is_premium_user'].iloc[0]))
total_spent = st.sidebar.number_input("Total Pengeluaran ($)", 0.0, 10000.0, float(baseline_df['total_spent'].iloc[0]))
lifetime = st.sidebar.number_input("Lifetime Value ($)", 0.0, 50000.0, float(baseline_df['lifetime_value'].iloc[0]))
ticket = st.sidebar.number_input("Jumlah Komplain", 0, 50, int(baseline_df['support_tickets'].iloc[0]))
satisfaction = st.sidebar.slider("Skor Kepuasan 1-5", 1, 5, int(baseline_df['satisfaction_score'].iloc[0]))
delay = st.sidebar.number_input("Hari Delay Pengiriman", 0, 30, int(baseline_df['delivery_delay_days'].iloc[0]))
refund = st.sidebar.selectbox("Pernah Refund?", ["Tidak", "Ya"], index=int(baseline_df['refund_requested'].iloc[0]))

st.sidebar.markdown("---")
demo_churn = st.sidebar.checkbox("🔥 Tes Pelanggan Berisiko Churn", help="Centang ini untuk isi nilai ekstrem yang memicu churn")

# ============================
# 4. BANGUN DATAFRAME INPUT
# ============================
st.title("🔮 Aplikasi Prediksi Churn Pelanggan")
st.write("Model dilatih dari dataset `Sales - Marketing customer dataset.csv`")

# KUNCI 1: Mulai dari baseline = median pelanggan normal, bukan 0
input_data = baseline_df.iloc[0].to_dict()

# KUNCI 2: Override dengan input user
input_data["age"] = age
input_data["total_spent"] = total_spent
input_data["lifetime_value"] = lifetime
input_data["support_tickets"] = ticket
input_data["satisfaction_score"] = satisfaction
input_data["delivery_delay_days"] = delay
input_data["refund_requested"] = 1 if refund == "Ya" else 0
input_data["is_premium_user"] = 1 if premium == "Ya" else 0

# KUNCI 3: Mode Demo = Set nilai ekstrem yang pasti bikin model predict churn
if demo_churn:
    input_data["support_tickets"] = 8 # Komplain banyak
    input_data["satisfaction_score"] = 1 # Puas rendah banget
    input_data["delivery_delay_days"] = 14 # Delay parah
    input_data["refund_requested"] = 1 # Pernah refund
    input_data["nps_score"] = 2 # NPS jelek
    input_data["lifetime_value"] = 80.0 # LTV rendah
    input_data["total_visits"] = 3 # Jarang visit

# KUNCI 4: Paksa urutan kolom 100% sama dengan saat training
input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

# ============================
# 5. PREDIKSI
# ============================
if st.button("Prediksi Status Churn"):

    input_scaled = scaler.transform(input_df)
    prob_churn = model.predict_proba(input_scaled)[0][1] # Prob kelas 1

    # Turunin threshold biar lebih sensitif. 0.5 -> 0.45
    prediction = 1 if prob_churn >= 0.45 else 0

    st.divider()
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"🚨 Pelanggan Diprediksi CHURN")
        else:
            st.success(f"✅ Pelanggan Diprediksi TIDAK CHURN")

    with col2:
        st.metric(label="Probabilitas Churn", value=f"{prob_churn*100:.2f}%", delta=f"{prob_churn*100-50:.1f}%" if demo_churn else None)

    st.progress(prob_churn) # Bar progress biar keliatan

    if prob_churn >= 0.45:
        st.warning("""
        ### Rekomendasi Aksi Retensi Segera
        - Hubungi pelanggan via CS/WhatsApp dalam 24 jam
        - Kirim voucher diskon 15% khusus churn risk
        - Eskalasi komplain ke tim senior
        """)
    else:
        st.info("### Rekomendasi: Pertahankan layanan & masukkan ke program loyalitas.")
