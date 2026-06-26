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
    'age', 'is_premium_user', 'total_visits', 'avg_session_time', 'pages_per_session', 
    'email_open_rate', 'email_click_rate', 'total_spent', 'avg_order_value', 'discount_used', 
    'support_tickets', 'refund_requested', 'delivery_delay_days', 'satisfaction_score', 
    'nps_score', 'marketing_spend_per_user', 'lifetime_value', 'last_3_month_purchase_freq', 
    'gender_Male', 'gender_Other', 'country_Germany', 'country_India', 'country_UK', 
    'country_USA', 'city_Delhi', 'city_Dhaka', 'city_Hamburg', 'city_London', 'city_Mumbai', 
    'city_New York', 'acquisition_channel_Facebook Ads', 'acquisition_channel_Google Ads', 
    'acquisition_channel_Organic', 'acquisition_channel_Referral', 'device_type_Mobile', 
    'device_type_Tablet', 'subscription_type_Monthly', 'coupon_code_REF10', 'coupon_code_SALE15', 
    'payment_method_Card', 'payment_method_PayPal', 'payment_method_SEPA', 'payment_method_UPI'
]

st.write("Isi semua fitur di bawah ini. Kategorikal: 0 atau 1.")

# 3. Input Form
input_data = {}
cols = st.columns(5)
for i, col in enumerate(FEATURE_COLUMNS):
    with cols[i % 5]:
        input_data[col] = st.number_input(col, value=0.0, format="%.4f", key=col)

if st.button('Prediksi Sekarang'):
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error(f'🚨 Hasil: PELANGGAN CHURN = {prediction}')
        st.warning("Alasan: Skor kepuasan rendah, banyak tiket komplain, dan minta refund. Disarankan tim retention segera menghubungi.")
    else:
        st.success(f'✅ Hasil: PELANGGAN TIDAK CHURN = {prediction}')
        st.info("Alasan: Pelanggan aktif, puas, dan tidak ada komplain.")
