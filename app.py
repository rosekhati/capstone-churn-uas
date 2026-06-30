import streamlit as st
import pandas as pd
import joblib

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

[data-testid="stSidebar"]{
    background-color:#EEF2F7;
}

.stButton>button{
    background-color:#4F46E5;
    color:white;
    border-radius:8px;
    width:220px;
    height:45px;
    font-size:16px;
    font-weight:bold;
}

h1{
    color:#1F2937;
}

</style>
""", unsafe_allow_html=True)

# ============================
# Load Model
# ============================
model = joblib.load("model_churn_terbaik.pkl")
scaler = joblib.load("scaler.pkl")

# ============================
# Feature Model
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

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Input Fitur Utama Pelanggan")

age = st.sidebar.number_input(
    "Usia",
    min_value=18,
    max_value=80,
    value=30
)

premium = st.sidebar.selectbox(
    "Premium User",
    ["Ya","Tidak"]
)

total_spent = st.sidebar.number_input(
    "Total Pengeluaran ($)",
    value=150.0
)

lifetime = st.sidebar.number_input(
    "Lifetime Value ($)",
    value=500.0
)

ticket = st.sidebar.number_input(
    "Jumlah Komplain",
    value=1
)

satisfaction = st.sidebar.slider(
    "Skor Kepuasan",
    1,
    5,
    4
)

# ==================================================
# HALAMAN UTAMA
# ==================================================

st.title("🔮 Aplikasi Prediksi Churn Pelanggan")

st.write("""
Aplikasi ini digunakan untuk memprediksi kemungkinan pelanggan
melakukan **Churn** berdasarkan fitur utama pelanggan.
""")

# ==================================================
# INPUT DATA
# ==================================================

# Semua fitur otomatis 0
input_data = dict.fromkeys(FEATURE_COLUMNS, 0)

# Isi fitur yang dipakai
input_data["age"] = age
input_data["total_spent"] = total_spent
input_data["lifetime_value"] = lifetime
input_data["support_tickets"] = ticket
input_data["satisfaction_score"] = satisfaction

if premium == "Ya":
    input_data["is_premium_user"] = 1
else:
    input_data["is_premium_user"] = 0

# ==================================================
# PREDIKSI
# ==================================================

if st.button("Prediksi Status Churn"):

    # Membuat DataFrame
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    # Probabilitas
    probability = model.predict_proba(input_scaled)[0]

    prob_not_churn = probability[0] * 100
    prob_churn = probability[1] * 100

    # Threshold keputusan
    threshold = 40

    st.divider()

    st.subheader("Hasil Prediksi:")

    # ===============================
    # OUTPUT
    # ===============================

    if prob_churn >= threshold:

        st.error(
            f"⚠️ Pelanggan Berpotensi CHURN (Berhenti Berlangganan) "
            f"dengan probabilitas {prob_churn:.2f}%"
        )

    else:

        st.success(
            f"✅ Pelanggan Diprediksi TIDAK CHURN "
            f"dengan probabilitas bertahan {prob_not_churn:.2f}%"
        )

    st.caption(
        f"Probabilitas churn mentah dari model: {prob_churn:.2f}% | "
        f"Threshold keputusan: {threshold}%"
    )

    st.divider()

    # ===============================
    # PROGRESS BAR
    # ===============================

    st.subheader("Visualisasi Probabilitas")

    st.progress(min(prob_churn/100, 1.0))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probabilitas Churn",
            f"{prob_churn:.2f}%"
        )

    with col2:
        st.metric(
            "Probabilitas Bertahan",
            f"{prob_not_churn:.2f}%"
        )

    st.divider()

    # ===============================
    # REKOMENDASI
    # ===============================

    if prob_churn >= threshold:

        st.warning("""
### 🚨 Rekomendasi

- Hubungi pelanggan secepatnya.
- Berikan promo atau voucher khusus.
- Tingkatkan kualitas pelayanan.
- Follow-up seluruh komplain pelanggan.
- Berikan program loyalitas.
""")

    else:

        st.info("""
### ✅ Rekomendasi

- Pertahankan kualitas pelayanan.
- Berikan reward loyalitas.
- Tingkatkan engagement pelanggan.
- Lakukan monitoring berkala.
""")
### Rekomendasi

- Hubungi pelanggan.
- Berikan promo khusus.
- Tingkatkan kualitas layanan.
- Tindak lanjuti komplain pelanggan.
        """)

    else:

        st.info("""
### Rekomendasi

- Pertahankan kualitas layanan.
- Berikan program loyalitas.
- Tingkatkan engagement pelanggan.
        """)
