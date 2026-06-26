# Customer Churn Prediction App

Aplikasi Machine Learning untuk memprediksi apakah pelanggan akan Churn atau tidak. 
Dibangun menggunakan Scikit-learn, Streamlit, dan di-deploy di Streamlit Cloud.

### **Tech Stack**
- **Model**: RandomForestClassifier / LogisticRegression 
- **Deployment**: Streamlit Cloud
- **Preprocessing**: StandardScaler + One-Hot Encoding

### **Fitur Utama**
- Input 43 fitur pelanggan sesuai hasil training
- Prediksi real-time: CHURN / TIDAK CHURN
- Model sudah di-scaling agar akurat

### **Link Aplikasi**
[https://capstone-churn-uas-dxkpkb45hwcpzqxbcg6e5u.streamlit.app/]

### **Cara Kerja Model**
Model dilatih dengan data 43 fitur. Jika `satisfaction_score` rendah, `support_tickets` tinggi, dan `refund_requested=1`, maka prediksi = CHURN.
