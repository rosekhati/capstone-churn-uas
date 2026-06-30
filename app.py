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
# 2. Load Model + Scaler Saja
# ============================
@st.cache_resource
def load_model():
    return joblib.load("model_churn_terbaik.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()

# ============================
# 3. Feature Model - WAJIB SAMA PERSIS URUTANNYA DENGAN SAAT TRAINING
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

# ============================
# 4. Bikin "Nilai Default Median" Manual
# -> Ini nggantiin file.csv. Isi 0 semua = aman buat scaler
# ============================
default_values = {col: 0.0 for col in FEATURE_COLUMNS}

# ==================================================
# 5. SIDEBAR INPUT
# ==================================================
st.sidebar.title("Input Fitur Pelanggan")
age = st.sidebar.number_input("Usia", 18, 80, 35)
premium = st.sidebar.selectbox("Premium User", ["Tidak","Ya"]) # Dibalik biar default 0
total_spent = st.sidebar.number_input("Total Pengeluaran ($)", 0.0, 10000.0, 0.0)
lifetime = st.sidebar.number_input("Lifetime Value ($)", 0.0, 50000.0, 0.0)
ticket = st.sidebar.number_input("Jumlah Komplain", 0, 50, 0)
satisfaction = st.sidebar.slider("Skor Kepuasan 1-5", 1, 5, 5) # Default 5 = aman
delay = st.sidebar.number_input("Hari Delay Pengiriman", 0, 30, 0)
refund = st.sidebar.selectbox("Pernah Refund?", ["Tidak", "Ya"])

st.sidebar.markdown("---")
demo_churn = st.sidebar.checkbox("Tes Pelanggan Berisiko Churn") # Checkbox lebih aman

# ==================================================
# 6. BANGUN DATAFRAME INPUT
# ==================================================
st.title("🔮 Aplikasi Prediksi Churn Pelanggan")

# Mulai dari default 0
input_data = default_values.copy()

# Isi yang dari user
input_data["age"] = age
input_data["total_spent"] = total_spent
input_data["lifetime_value"] = lifetime
input_data["support_tickets"] = ticket
input_data["satisfaction_score"] = satisfaction
input_data["delivery_delay_days"] = delay
input_data["refund_requested"] = 1 if refund == "Ya" else 0
input_data["is_premium_user"] = 1 if premium == "Ya" else 0

# Override kalau mode demo diaktifin -> Ini pasti churn
if demo_churn:
    input_data["support_tickets"] = 10
    input_data["satisfaction_score"] = 1
    input_data["delivery_delay_days"] = 15
    input_data["refund_requested"] = 1
    input_data["total_visits"] = 2
    input_data["lifetime_value"] = 50.0

# KUNCI: Bikin DataFrame dan paksa urutan kolomnya sama persis
input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

# ==================================================
# 7. PREDIKSI
# ==================================================
if st.button("Prediksi Status Churn"):

    try:
        input_scaled = scaler.transform(input_df)
        prob_churn = model.predict_proba(input_scaled)[0][1] # Ambil prob kelas 1 = churn
        prediction = 1 if prob_churn >= 0.5 else 0 # Threshold 0.5

        st.divider()
        st.subheader("Hasil Prediksi")

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"🚨 Pelanggan Diprediksi CHURN")
            else:
                st.success(f"✅ Pelanggan Diprediksi TIDAK CHURN")

        with col2:
            st.metric(label="Probabilitas Churn", value=f"{prob_churn*100:.2f}%")

        if prob_churn >= 0.5:
            st.warning("### Rekomendasi: Hubungi, beri promo retensi, tindak lanjuti komplain.")
        else:
            st.info("### Rekomendasi: Pertahankan layanan & berikan loyalty.")

    except Exception as e:
        st.error(f"Terjadi Error: {e}")
        st.info("Cek: 1. Apakah `FEATURE_COLUMNS` urutannya 100% sama dengan saat training? 2. Apakah `scaler.pkl` dan `model.pkl` satu paket?")
