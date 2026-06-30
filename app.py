import streamlit as st
import pandas as pd
import joblib

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Prediksi Churn",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Prediksi Churn Pelanggan")
st.write("UAS Bengkel Koding Data Science - Rosekhati - A11.2023.15496")

# =========================
# Load Model
# =========================
model = joblib.load("model_churn_terbaik.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Nama Fitur
# =========================
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

st.write("### Masukkan Nilai Fitur")

# =========================
# Input
# =========================
input_data = {}

cols = st.columns(5)

for i, col in enumerate(FEATURE_COLUMNS):
    with cols[i % 5]:
        input_data[col] = st.number_input(
            label=col,
            value=0.0,
            format="%.4f",
            key=col
        )

# =========================
# Prediksi
# =========================
if st.button("Prediksi Sekarang"):

    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    # Probabilitas
    probability = model.predict_proba(input_scaled)[0]

    prob_not_churn = probability[0] * 100
    prob_churn = probability[1] * 100

    # Threshold
    threshold = 40

    st.markdown("---")
    st.header("Hasil Prediksi")

    if prob_churn >= threshold:

        st.error(
            f"⚠️ Pelanggan Berpotensi CHURN (Berhenti Berlangganan) "
            f"dengan probabilitas {prob_churn:.2f}%"
        )

        st.warning("""
### Rekomendasi

- Hubungi pelanggan secepatnya.

- Berikan promo atau diskon.

- Tingkatkan kualitas pelayanan.

- Tindak lanjuti seluruh komplain pelanggan.
""")

    else:

        st.success(
            f"✅ Pelanggan Diprediksi TIDAK CHURN "
            f"dengan probabilitas bertahan {prob_not_churn:.2f}%"
        )

        st.info("""
### Rekomendasi

- Pertahankan kualitas pelayanan.

- Berikan program loyalitas.

- Tingkatkan engagement pelanggan.
""")

    # Progress Bar
    st.subheader("Probabilitas Churn")

    st.progress(min(prob_churn / 100, 1.0))

    col1, col2 = st.columns(2)

    col1.metric(
        "Probabilitas Churn",
        f"{prob_churn:.2f}%"
    )

    col2.metric(
        "Probabilitas Tidak Churn",
        f"{prob_not_churn:.2f}%"
    )

    st.caption(
        f"Threshold keputusan = {threshold}% | "
        f"Prediksi model = {prediction}"
    )
