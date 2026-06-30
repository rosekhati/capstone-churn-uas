import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="🔮",
    layout="wide"
)

# ============================
# CSS
# ============================
st.markdown("""
<style>
[data-testid="stSidebar"]{ background-color:#EEF2F7; }
.stButton>button{
    background-color:#4F46E5; color:white; border-radius:8px;
    width:220px; height:45px; font-size:16px; font-weight:bold;
}
h1{ color:#1F2937; }
</style>
""", unsafe_allow_html=True)

# ============================
# Load Model, Scaler, dan Default Data
# ============================
@st.cache_resource # biar nggak load ulang tiap klik
def load_artifacts():
    model = joblib.load("model_churn_terbaik.pkl")
    scaler = joblib.load("scaler.pkl")
    # PENTING: Simpan juga 1 baris data training saat fit scaler
    # untuk jadi baseline. Kalau nggak ada, pakai median 0.
    try:
        baseline_df = pd.read_csv("baseline_churn_input.csv") # Simpan ini dari notebook
    except:
        baseline_df = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS) # Fallback
    return model, scaler, baseline_df

# ============================
# Feature Model - Harus urutannya sama persis saat training
# ============================
FEATURE_COLUMNS = [
    'age', 'is_premium_user', 'total_visits', 'avg_session_time',
    'pages_per_session', 'email_open_rate', 'email_click_rate',
    'total_spent', 'avg_order_value', 'discount_used',
    'support_tickets', 'refund_requested', 'delivery_delay_days',
    'satisfaction_score', 'nps_score',
    'marketing_spend_per_user', 'lifetime_value',
    'last_3_month_purchase_freq',
    'gender_Male', 'gender_Other',
    'country_Germany', 'country_India',
    'country_UK', 'country_USA',
    'city_Delhi', 'city_Dhaka',
    'city_Hamburg', 'city_London',
    'city_Mumbai', 'city_New York',
    'acquisition_channel_Facebook Ads',
    'acquisition_channel_Google Ads',
    'acquisition_channel_Organic',
    'acquisition_channel_Referral',
    'device_type_Mobile',
    'device_type_Tablet',
    'subscription_type_Monthly',
    'coupon_code_REF10',
    'coupon_code_SALE15',
    'payment_method_Card',
    'payment_method_PayPal',
    'payment_method_SEPA',
    'payment_method_UPI'
]

model, scaler, baseline_df = load_artifacts()

# ==================================================
# SIDEBAR - Input yang ngaruh ke Churn
# ==================================================
st.sidebar.title("Input Fitur Utama Pelanggan")

age = st.sidebar.number_input("Usia", 18, 80, 30)
premium = st.sidebar.selectbox("Premium User", ["Ya","Tidak"])
total_spent = st.sidebar.number_input("Total Pengeluaran ($)", 0.0, 10000.0, 150.0)
lifetime = st.sidebar.number_input("Lifetime Value ($)", 0.0, 50000.0, 500.0)
ticket = st.sidebar.number_input("Jumlah Komplain", 0, 50, 1)
satisfaction = st.sidebar.slider("Skor Kepuasan 1-5", 1, 5, 4)

# Tambah 2 input yang paling ngaruh ke churn biar bisa ke-trigger
delay = st.sidebar.number_input("Hari Delay Pengiriman", 0, 30, 0)
refund = st.sidebar.selectbox("Pernah Refund?", ["Tidak", "Ya"])

# ==================================================
# HALAMAN UTAMA
# ==================================================
st.title("🔮 Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi ini memprediksi kemungkinan pelanggan melakukan **Churn**.")

# ==================================================
# INPUT DATA
# ==================================================
# 1. Mulai dari baseline, bukan 0 semua. Ini kuncinya.
input_data = baseline_df.iloc[0].to_dict()

# 2. Override dengan input user
input_data["age"] = age
input_data["total_spent"] = total_spent
input_data["lifetime_value"] = lifetime
input_data["support_tickets"] = ticket
input_data["satisfaction_score"] = satisfaction
input_data["delivery_delay_days"] = delay
input_data["refund_requested"] = 1 if refund == "Ya" else 0
input_data["is_premium_user"] = 1 if premium == "Ya" else 0

# 3. Contoh bikin pelanggan "berpotensi churn": komplain tinggi + puas rendah
st.sidebar.markdown("---")
if st.sidebar.button("Coba Mode Pelanggan Berisiko"):
    st.sidebar.session_state['demo'] = True

if st.sidebar.session_state.get('demo', False):
    input_data["support_tickets"] = 5
    input_data["satisfaction_score"] = 1
    input_data["delivery_delay_days"] = 7
    input_data["refund_requested"] = 1

# Pastikan urutan kolom sama persis
input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

# ==================================================
# PREDIKSI
# ==================================================
if st.button("Prediksi Status Churn"):

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"🚨 Pelanggan Diprediksi CHURN")
        else:
            st.success(f"✅ Pelanggan Diprediksi TIDAK CHURN")

    with col2:
        st.metric(
            "Probabilitas Churn",
            f"{probability[1]*100:.2f}%"
        )

    st.divider()
    # Threshold bisa kamu turunin kalau mau lebih sensitif
    if probability[1] > 0.4: # Default model 0.5. Turunin ke 0.4 biar lebih sering warning
        st.warning("""
### Rekomendasi Aksi Cepat
- Hubungi pelanggan segera.
- Berikan voucher/promo retensi.
- Eskalasi komplain ke CS senior.
        """)
    else:
        st.info("""
### Rekomendasi
- Pertahankan kualitas layanan.
- Berikan program loyalitas.
        """)
