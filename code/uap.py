import streamlit as st
import joblib
import gdown
import os
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

# URL model dan dataset
files = [
    ("https://drive.google.com/uc?id=1shbUbDl_eanPjNG42lwA4zVi4ZrdZm1F", "rf_model.pkl"),
    ("https://drive.google.com/uc?id=1jBTkz7Vqv8nrVMGgY0PGtRF8O-p1-WRA", "my_model.keras"),
]

# URL dataset
dataset_url = "https://drive.google.com/uc?id=1_oNqybyeIxd3lnZwD_esMTNSmqgB_hsI"
dataset_path = "hotel_bookings.csv"

# Memuat dataset
data = pd.read_csv(dataset_path)

# Memuat model
rf_model = joblib.load("rf_model.pkl")
my_model = load_model("my_model.keras")

# Fungsi prediksi dengan format input yang benar
def make_prediction(model_choice, input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Pastikan input berupa array 2D
    if model_choice == "Random Forest":
        try:
            prediction_proba = rf_model.predict_proba(input_data)
            return prediction_proba[0] if prediction_proba is not None else None
        except Exception as e:
            st.error(f"Error in Random Forest prediction: {str(e)}")
            return None
    elif model_choice == "Feedforward Neural Network":
        try:
            prediction_proba = my_model.predict(input_data)  
            prob_tidak_batal = prediction_proba[0][0]  # Probabilitas 'Tidak Batal'
            prob_batal = 1 - prob_tidak_batal  # Probabilitas 'Batal'
            return [prob_tidak_batal, prob_batal]
        except Exception as e:
            st.error(f"Error in Feedforward Neural Network prediction: {str(e)}")
            return None

# Header dan Deskripsi
st.title('Prediksi Pembatalan Pemesanan Hotel')
st.image("src/uap_ml/assets/hotel.jpeg", use_container_width=True)  # Gambar Header
st.write("Aplikasi ini memprediksi kemungkinan pembatalan pemesanan hotel.")

# Sidebar Input
with st.sidebar:
    st.header("Masukkan Informasi Pemesanan")
    
    # 1. Fitur kategori
    hotel_mapping = {0: "Resort Hotel", 1: "City Hotel"}
    hotel_number = st.selectbox("1. Pilih Jenis Hotel", [0, 1])
    hotel_selected = hotel_mapping[hotel_number]

    # 2. Fitur numerik dan kategori lainnya
    lead_time = st.number_input("2. Lead Time (hari)", min_value=0, max_value=700, step=1)
    arrival_date_year = st.number_input("3. Tahun Kedatangan", min_value=2020, max_value=2030, step=1, value=2024)
    arrival_date_month = st.selectbox("4. Pilih Bulan Kedatangan", list(range(1, 13)))
    arrival_month_bin = 1 if arrival_date_month in [6, 7, 8] else 0
    arrival_date_week_number = st.number_input("5. Minggu dalam Tahun", min_value=1, max_value=52, step=1)
    arrival_date_day_of_month = st.number_input("6. Hari dalam Bulan", min_value=1, max_value=31, step=1)

    num_weekend_nights = st.number_input("7. Jumlah Malam Akhir Pekan", min_value=0, max_value=10, step=1)
    num_week_nights = st.number_input("8. Jumlah Malam Hari Kerja", min_value=0, max_value=30, step=1)
    total_stay = num_weekend_nights + num_week_nights
    only_weekday_stay = 1 if num_weekend_nights == 0 else 0
    adults = st.number_input("9. Jumlah Dewasa", min_value=1, max_value=10, step=1)
    have_children = 1 if st.number_input("10. Jumlah Anak", min_value=0, max_value=10, step=1) > 0 else 0
    babies = st.number_input("11. Jumlah Bayi", min_value=0, max_value=10, step=1)

    # 12. Paket Makanan
    meal_mapping = {0: "Tanpa Paket", 1: "Paket A", 2: "Paket B"}
    meal_number = st.selectbox("12. Paket Makanan", list(meal_mapping.keys()))
    meal_selected = meal_mapping[meal_number]

    # 13. Pilih Negara
    country_mapping = {
        "PRT": 1,  # Portugal
        "ESP": 2,  # Spain
        "FRA": 3,  # France
        "ITA": 4,  # Italy
        "USA": 5,  # USA
        "GER": 6,  # Germany
    }
    country_number = st.selectbox("13. Pilih Negara", list(country_mapping.keys()))
    country_encoded = country_mapping[country_number]

    # 14. Market Segment
    market_segment = st.selectbox("14. Segmentasi Pasar", ["Online", "Offline", "Corporate", "Direct", "TA/TO", "Complementary", "Undefined"])
    market_segment_bin = {
        "Online": 0, "Offline": 1, "Corporate": 2, "Direct": 3,
        "TA/TO": 4, "Complementary": 5, "Undefined": 6
    }[market_segment]

    # 15. Channel Distribusi
    distribution_channel_mapping = {
        "Online": 0, "Offline": 1, "Corporate": 2
    }
    distribution_channel = st.selectbox("15. Channel Distribusi", ["Online", "Offline", "Corporate"])
    distribution_channel_bin = distribution_channel_mapping[distribution_channel]

    # 16. Apakah Tamu Berulang
    is_repeated_guest_mapping = {"Tidak": 0, "Ya": 1}
    repeated_guest_choice = st.selectbox("16. Apakah Tamu Berulang?", list(is_repeated_guest_mapping.keys()))
    is_repeated_guest = is_repeated_guest_mapping[repeated_guest_choice]

    # 17. Pembatalan Sebelumnya dan lainnya
    previous_cancellations = st.number_input("17. Pembatalan Sebelumnya", min_value=0, max_value=10, step=1)
    previous_bookings_not_canceled = st.number_input("18. Pemesanan Sebelumnya (Tidak Dibatalkan)", min_value=0, max_value=10, step=1)

    # 18. Kamar Sama Ditugaskan
    same_room_mapping = {"Tidak": 0, "Ya": 1}
    same_room_choice = st.selectbox("19. Kamar Sama Ditugaskan?", list(same_room_mapping.keys()))
    same_room_assigned = same_room_mapping[same_room_choice]

    # 19. Input lainnya
    agent = "9"  # Default, atau input jika relevan
    days_in_waiting_list = st.number_input("20. Hari dalam Daftar Tunggu", min_value=0, max_value=365, step=1)

    # 20. Tipe Pelanggan
    customer_type_mapping = {"Biasa": 0, "Grup": 1, "Loyal": 2}
    customer_type_choice = st.selectbox("21. Tipe Pelanggan", list(customer_type_mapping.keys()))
    customer_type_bin = customer_type_mapping[customer_type_choice]

    # 21. ADR dan Parkir
    adr = st.number_input("22. ADR (Harga per Malam)", min_value=0.0, step=1.0)
    parking_mapping = {"Tidak": 0, "Ya": 1}
    parking_choice = st.selectbox("23. Butuh Parkir?", list(parking_mapping.keys()))
    is_parking_required = parking_mapping[parking_choice]

    total_of_special_requests = st.number_input("24. Jumlah Permintaan Khusus", min_value=0, max_value=10, step=1)

    # Format input_data
    input_data = [
        hotel_number, lead_time, arrival_date_year, arrival_date_month, arrival_month_bin, 
        arrival_date_week_number, arrival_date_day_of_month, total_stay, only_weekday_stay, 
        adults, have_children, babies, meal_number, country_encoded, 1, 
        distribution_channel_bin, is_repeated_guest, previous_cancellations, 
        previous_bookings_not_canceled, same_room_assigned, agent, 
        days_in_waiting_list, customer_type_bin, adr, is_parking_required, 
        total_of_special_requests, market_segment_bin
    ]
    
   # Pemilih model di atas tombol prediksi
st.markdown("### Pilih Model")
model_choice = st.selectbox(
    "Pilih model prediksi yang ingin digunakan:",
    ["Random Forest", "Feedforward Neural Network (FNN)"],
    help="Pilih model untuk melakukan prediksi kemungkinan pembatalan pemesanan."
)

# Prediksi
if st.button('Prediksi', use_container_width=True):
    prediction_proba = make_prediction(model_choice, input_data)

    if prediction_proba is not None:
        result = 'Batal' if prediction_proba[1] > prediction_proba[0] else 'Tidak Batal'

        st.subheader("Hasil Prediksi")
        st.success(f"Kemungkinan: {result}", icon="✅")
        st.markdown(f"*Model yang digunakan*: {model_choice}")
        st.markdown(f"*Probabilitas Batal*: {prediction_proba[1]*100:.2f}%")
        st.markdown(f"*Probabilitas Tidak Batal*: {prediction_proba[0]*100:.2f}%")
    else:
        st.error("Prediksi gagal. Silakan coba lagi.", icon="❌")
