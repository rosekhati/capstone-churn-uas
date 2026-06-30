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
# 2. Load Model + Scaler
# ============================
@st.cache_resource
def load_model():
    return joblib.load("model_churn_terbaik.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()

# ============================
# 3. Feature Model - WAJIB SAMA PERSIS URUTANNYA
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
# 4. KUNCI: Nilai Median Normal. Ini baseline "pelanggan aman"
# -> Ambil dari df.describe() kamu. Jangan 0 semua.
# ============================
BASELINE_VALUES = {
    'age': 35, 'is_premium_user': 0, 'total_visits': 12, 'avg_session_time': 7.5,
    'pages_per_session': 3.8, 'email_open_rate': 0.22, 'email_click_rate': 0.04,
    'total_spent': 320.0, 'avg_order_value': 52.0, 'discount_used': 1,
    'support_tickets': 0, 'refund_requested': 0, 'delivery_delay_days': 2,
    'satisfaction_score': 4, 'nps_score': 7,
    'marketing_spend_per_user': 10.0, 'lifetime_value': 950.0,
    'last_3_month_purchase_freq': 2,
    'gender_Male': 0, 'gender_Other': 0,
    'country_Germany': 0, 'country_India': 1, 'country_UK': 0, 'country_USA': 0,
    'city_Delhi': 1, 'city_Dhaka': 0, 'city_Hamburg': 0, 'city_London': 0, 'city_Mumbai': 0, 'city_New York': 0,
    'acquisition_channel_Facebook Ads': 0, 'acquisition_channel_Google Ads': 0,
    'acquisition_channel_Organic': 1, 'acquisition_channel_Referral': 0,
    'device_type_Mobile': 1, 'device_type_Tablet': 0,
    'subscription_type_Monthly': 1,
    'coupon_code_REF10': 0, 'coupon_code_SALE15': 0,
    'payment_method_Card': 1, 'payment_method_PayPal': 0, 'payment_method_SEPA': 0, 'payment_method_UPI': 0
}

# ==================================================
# 5. SIDEBAR INPUT - HANYA 6 FITUR PALING NGARUH KE CHURN
# ==================================================
st.sidebar.title("Input Fitur Pelanggan")
st.sidebar.caption("Isi 6 fitur ini. Sisanya pakai nilai normal.")

col1, col2 = st.sidebar.columns(2)
with col1:
    ticket = st.number_input("Jumlah Komplain", 0, 50, 0, help=">3 = risiko tinggi")
    delay = st.number_input("Hari Delay Kirim", 0, 30, 2, help=">7 = risiko tinggi")
    lifetime = st.number_input("Lifetime Value ($)", 0.0, 50000.0, 950.0, step=50.0, help="<200 = risiko tinggi")
with col2:
    satisfaction = st.slider("Skor Kepuasan 1-5", 1, 5, 4, help="<=2 = risiko tinggi")
    refund = st.selectbox("Pernah Refund?", ["Tidak", "Ya"])
    visits = st.number_input("Total Kunjungan App", 0, 200, 12, help="<5 = risiko tinggi")

# ==================================================
# 6. BANGUN DATAFRAME INPUT DARI INPUT USER
# ==================================================
st.title("🔮 Aplikasi Prediksi Churn Pelanggan")
st.write("Hasil `CHURN` murni dari 6 fitur di atas. Coba ubah nilainya.")

# 1. Mulai dari baseline normal
input_data = BASELINE_VALUES.copy()

# 2. Override HANYA dengan input user
input_data["support_tickets"] = ticket
input_data["delivery_delay_days"] = delay
input_data["lifetime_value"] = lifetime
input_data["satisfaction_score"] = satisfaction
input_data["refund_requested"] = 1 if refund == "Ya" else 0
input_data["total_visits"] = visits

# 3. Paksa urutan kolom sama persis
input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

# ==================================================
# 7. PREDIKSI
# ==================================================
if st.button("Prediksi Status Churn"):

    input_scaled = scaler.transform(input_df)
    prob_churn = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob_churn >= 0.5 else 0 # Balik ke 0.5 biar adil

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
        st.progress(prob_churn)

    st.caption(f"Input kamu: Komplain={ticket}, Kepuasan={satisfaction}, Delay={delay} hari, Refund={'Ya' if refund=='Ya' else 'Tidak'}, LTV=${lifetime}, Visits={visits}")

    if prob_churn >= 0.5:
        st.warning("### Rekomendasi: Segera lakukan retensi. Pelanggan berisiko tinggi cabut.")
    else:
        st.info("### Rekomendasi: Pertahankan. Pelanggan masih aman.")
