import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN (Wajib di paling atas)
# ==========================================
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS UNTUK TAMPILAN MENARIK
# ==========================================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        background-color: #4F46E5;
        color: white;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        border: 2px solid #4338CA;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1F2937;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h3 {
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI PEMUATAN MODEL DENGAN CACHE
# ==========================================
@st.cache_resource
def load_model_and_scaler():
    try:
        # Gunakan try-except untuk berjaga-jaga jika file belum ada
        model = joblib.load('best_rf_churn_model_randomsearchCV.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("⚠️ File model (.pkl) tidak ditemukan. Pastikan Anda sudah menjalankan script training dan menyimpan modelnya.")
        return None, None, None

model, scaler, model_columns = load_model_and_scaler()

# ==========================================
# 4. ANTARMUKA UTAMA (HEADER & DESKRIPSI)
# ==========================================
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3121/3121693.png", width=80) # Ikon dummy representasi data
with col2:
    st.title('Sistem Cerdas Prediksi Churn Pelanggan')
    st.markdown('*Project UAS Bengkel Koding - Data Science | Oleh: Yuda Fuji Hermanysah | A11.2023.14908*')

st.markdown("---")
st.write("Aplikasi ini menggunakan model **Machine Learning (Random Forest)** yang telah dioptimasi untuk memprediksi probabilitas seorang pelanggan akan meninggalkan layanan (Churn) atau tetap berlangganan (Retained).")

# ==========================================
# 5. SIDEBAR: INPUT DATA PELANGGAN
# ==========================================
st.sidebar.header('📝 Masukkan Data Pelanggan')
st.sidebar.markdown("Sesuaikan nilai di bawah ini untuk melihat hasil prediksi.")

def user_input_features():
    # Menggunakan container untuk merapikan sidebar
    with st.sidebar.container():
        st.markdown("### 📊 Fitur Utama (Top 10)")
        # Masukkan 10 fitur dari grafik Feature Importances
        total_spent = st.number_input('Total Pengeluaran ($)', min_value=0.0, value=1500.0, step=50.0)
        satisfaction_score = st.slider('Skor Kepuasan (1-5)', 1, 5, 3)
        support_tickets = st.number_input('Jumlah Support Tickets', min_value=0, value=2, step=1)
        avg_session_time = st.number_input('Rata-rata Waktu Sesi (Menit)', min_value=0.0, value=8.5, step=0.5)
        lifetime_value = st.number_input('Lifetime Value ($)', min_value=0.0, value=5000.0, step=100.0)
        avg_order_value = st.number_input('Rata-rata Nilai Order ($)', min_value=0.0, value=100.0, step=10.0)
        marketing_spend = st.number_input('Marketing Spend per User ($)', min_value=0.0, value=20.0, step=5.0)
        pages_per_session = st.number_input('Halaman per Sesi', min_value=1, value=5, step=1)
        email_open_rate = st.slider('Email Open Rate', 0.0, 1.0, 0.5)
        email_click_rate = st.slider('Email Click Rate', 0.0, 1.0, 0.1)
        
        st.markdown("### ⚙️ Fitur Tambahan")
        age = st.slider('Usia', 18, 80, 30)
        is_premium_user = st.radio('Pengguna Premium?', ('Tidak (0)', 'Ya (1)'))
        is_premium_val = 1 if 'Ya' in is_premium_user else 0
        payment_method = st.selectbox('Metode Pembayaran', ('Credit Card', 'PayPal', 'Bank Transfer', 'UPI'))
        
    data = {
        'total_spent': total_spent,
        'satisfaction_score': satisfaction_score,
        'support_tickets': support_tickets,
        'avg_session_time': avg_session_time,
        'lifetime_value': lifetime_value,
        'avg_order_value': avg_order_value,
        'marketing_spend_per_user': marketing_spend,
        'pages_per_session': pages_per_session,
        'email_open_rate': email_open_rate,
        'email_click_rate': email_click_rate,
        'age': age,
        'is_premium_user': is_premium_val,
        f'payment_method_{payment_method}': 1 # Simulasi One-Hot Encoding
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ==========================================
# ==========================================
# 6. TAMPILAN DATA INPUT (SUMMARY)
# ==========================================
st.subheader("📋 Ringkasan Profil Pelanggan")

# Menampilkan 4 metrik utama, diurutkan dari yang paling memengaruhi Churn
m1, m2, m3, m4 = st.columns(4)

# 1. Skor Kepuasan (Prioritas Pertama)
sat_score = int(input_df['satisfaction_score'][0])
# Logika: Jika skor di bawah 3, munculkan peringatan merah
sat_delta = "Aman" if sat_score >= 3 else "- Kritis"
sat_color = "normal" if sat_score >= 3 else "inverse"

m1.metric(label="⭐ Skor Kepuasan", 
          value=f"{sat_score} / 5", 
          delta=sat_delta, 
          delta_color=sat_color)

# 2. Support Tickets (Prioritas Kedua)
tickets = int(input_df['support_tickets'][0])
# Logika: Jika tiket keluhan lebih dari 3, munculkan peringatan merah
ticket_delta = "- Banyak Keluhan" if tickets > 3 else "Wajar"
ticket_color = "inverse" if tickets > 3 else "normal"

m2.metric(label="🎫 Support Tickets", 
          value=f"{tickets} Tiket", 
          delta=ticket_delta, 
          delta_color=ticket_color)

# 3. Total Pengeluaran
m3.metric(label="💰 Total Pengeluaran", 
          value=f"${input_df['total_spent'][0]:,.2f}")

# 4. Lifetime Value (LTV)
m4.metric(label="💎 Lifetime Value", 
          value=f"${input_df['lifetime_value'][0]:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# 6.5 PENJELASAN FITUR 
# ==========================================
with st.expander("📌 Klik untuk melihat Penjelasan Fitur"):
    feature_desc = {
        "Total Pengeluaran ($)": "Total keseluruhan uang yang telah dibelanjakan oleh pelanggan.",
        "Skor Kepuasan (1-5)": "Tingkat kepuasan pelanggan terhadap layanan (1 = Sangat Kecewa, 5 = Sangat Puas). Fitur terpenting penentu churn.",
        "Jumlah Support Tickets": "Berapa kali pelanggan mengajukan komplain atau meminta bantuan teknis.",
        "Rata-rata Waktu Sesi (Menit)": "Rata-rata durasi (lama waktu) pelanggan mengakses aplikasi dalam sekali kunjungan.",
        "Lifetime Value ($)": "Estimasi total nilai keuntungan yang diberikan pelanggan selama ia berlangganan.",
        "Rata-rata Nilai Order ($)": "Rata-rata nominal uang yang dihabiskan pelanggan dalam satu kali transaksi.",
        "Marketing Spend per User ($)": "Biaya pemasaran (promo/iklan) yang dialokasikan perusahaan untuk pelanggan ini.",
        "Halaman per Sesi": "Jumlah rata-rata halaman yang dibuka oleh pelanggan dalam satu kali akses.",
        "Email Open Rate": "Tingkat persentase pelanggan membuka email promosi yang dikirimkan perusahaan.",
        "Email Click Rate": "Tingkat persentase pelanggan mengklik tautan/link yang ada di dalam email promosi.",
        "Usia": "Umur pelanggan saat ini.",
        "Pengguna Premium?": "Status apakah pelanggan menggunakan paket berbayar (Ya) atau gratis (Tidak).",
        "Metode Pembayaran": "Cara utama pelanggan melakukan pembayaran layanan."
    }
    
    desc_df = pd.DataFrame(list(feature_desc.items()), columns=["Fitur", "Keterangan"])
    st.dataframe(desc_df, width='stretch', hide_index=True)
# -----------------------------------------------

# ==========================================
# 7. PROSES PREDIKSI & VISUALISASI HASIL
# ==========================================
# Tombol prediksi di tengah
_, center_btn, _ = st.columns([1, 2, 1])

with center_btn:
    predict_button = st.button('🔍 Analisis & Prediksi Churn', width='stretch')

if predict_button:
    if model is not None and scaler is not None:
        with st.spinner('Memproses data menggunakan Random Forest...'):
            try:
                # 7a. Preprocessing Input agar sesuai dengan training data
                # Kita langsung menggunakan input_df dan menyesuaikan kolomnya
                input_processed = input_df.reindex(columns=model_columns, fill_value=0)
                
                # Scaling
                input_scaled = scaler.transform(input_processed)
                
                # 7b. Prediksi
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                churn_probability = prediction_proba[0][1] * 100
                retain_probability = prediction_proba[0][0] * 100
                
                st.markdown("---")
                st.subheader("🎯 Hasil Prediksi Model")
                
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    # Menampilkan pesan berdasarkan hasil
                    if prediction[0] == 1:
                        st.error("### ⚠️ PELANGGAN BERISIKO TINGGI CHURN")
                        st.markdown(f"Model mendeteksi pelanggan ini kemungkinan besar akan **berhenti** berlangganan dengan tingkat probabilitas **{churn_probability:.1f}%**.")
                        st.markdown("**Rekomendasi Tindakan:**")
                        st.markdown("- Berikan penawaran diskon khusus (Retensi).")
                        st.markdown("- Kirimkan email *follow-up* personal.")
                        st.markdown("- Cek apakah ada keluhan (Support Tickets) yang belum diselesaikan.")
                    else:
                        st.success("### ✅ PELANGGAN CENDERUNG SETIA (RETAINED)")
                        st.markdown(f"Model mendeteksi pelanggan ini kemungkinan besar akan **tetap** berlangganan dengan tingkat probabilitas **{retain_probability:.1f}%**.")
                        st.markdown("**Rekomendasi Tindakan:**")
                        st.markdown("- Tawarkan program loyalitas (*Up-selling/Cross-selling*).")
                        st.markdown("- Pertahankan kualitas pelayanan saat ini.")
                
                with res_col2:
                    # Membuat Gauge Chart yang sangat menarik menggunakan Plotly
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = churn_probability,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risiko Churn (%)", 'font': {'size': 24}},
                        number = {'suffix': "%", 'font': {'size': 40}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "rgba(0,0,0,0)"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': "#10B981"},    # Hijau (Aman)
                                {'range': [30, 70], 'color': "#F59E0B"},   # Kuning (Hati-hati)
                                {'range': [70, 100], 'color': "#EF4444"}], # Merah (Bahaya)
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': churn_probability}
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, width='stretch')
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam memproses input: {e}")
