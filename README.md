# Prediksi Pembatalan Pemesanan Hotel
-------------------------------
## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan aplikasi berbasis web yang dapat memprediksi pembatalan pemesanan hotel. Prediksi dilakukan berdasarkan data demografis dan preferensi individu, seperti jenis hotel, tahun kedatangan, jumlah dewasa, paket makanan, segmentasi pasar, apakah tamu berulang, tipe pelanggan dan lainnya. Aplikasi ini menyediakan opsi untuk memilih model prediksi, seperti Random Forest dan Feedforward Neural Network (FNN), sehingga pengguna dapat membandingkan performa dari berbagai algoritma. 

-------------------------------------
## Tujuan Pengembangan
1. Memberikan Prediksi Akurat Pembatalan Pemesanan apakah seorang pelanggan akan membatalkan pemesanan hotel berdasarkan data yang diberikan.
2. Memungkinkan eksplorasi performa model yang berbeda.
3. Menyediakan visualisasi data dan analisis yang interaktif untuk meningkatkan pengalaman pengguna.

--------------------------------
## Dataset
Dataset yang digunakan adalah [hotels](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
Dataset ini memili jumlah data sebanyak 119,390  dan 32 fitur.Berikut contoh beberapa fitur nya

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| hotel                            | Jenis hotel (City Hotel atau Resort Hotel)                                  |
| is_canceled                      | Status pembatalan (0 = tidak dibatalkan, 1 = dibatalkan)                    |
| lead_time                        | Waktu antara pemesanan dan tanggal kedatangan (dalam hari)                  |
| arrival_date_year                | Tahun kedatangan                                                            |
| arrival_date_month               | Bulan kedatangan                                                            |
| arrival_date_week_number         | Nomor minggu kedatangan                                                     |
| arrival_date_day_of_month        | Hari dalam bulan kedatangan                                                 |
| stays_in_weekend_nights          | Jumlah malam akhir pekan yang dipesan                                       |
| stays_in_week_nights             | Jumlah malam minggu yang dipesan                                            |
| adults                           | Jumlah tamu dewasa                                                           |
| country                          | Negara asal tamu                                                            |
| market_segment                   | Segmen pasar pemesanan                                                      |
| distribution_channel             | Saluran distribusi pemesanan                                                |
| is_repeated_guest                | Status tamu berulang (0 = bukan tamu berulang, 1 = tamu berulang)          |
| previous_cancellations           | Jumlah pembatalan sebelumnya oleh tamu yang sama                            |
| previous_bookings_not_canceled   | Jumlah pemesanan sebelumnya yang tidak dibatalkan oleh tamu yang sama       |
| reserved_room_type               | Tipe kamar yang dipesan                                                     |
| assigned_room_type               | Tipe kamar yang ditugaskan                                                  |
| booking_changes                  | Jumlah perubahan pada pemesanan                                            |
| deposit_type                     | Jenis deposit yang dibayar                                                  |
| agent                            | ID agen perjalanan yang membuat pemesanan                                  |
| company                          | ID perusahaan yang membuat pemesanan                                        |
| days_in_waiting_list             | Jumlah hari dalam daftar tunggu sebelum pemesanan dikonfirmasi              |
| customer_type                    | Tipe pelanggan (misalnya, Transient, Contract, Group)                       |
| adr                              | Tarif harian rata-rata                                                      |
| required_car_parking_spaces      | Jumlah ruang parkir mobil yang dibutuhkan                                   |
| total_of_special_requests        | Jumlah permintaan khusus yang dibuat oleh tamu                              |
| reservation_status               | Status pemesanan (misalnya, Check-Out, Canceled)                            |
| reservation_status_date          | Tanggal status pemesanan diperbarui                                         |
--------------------
## Langkah Instalasi
1. Clone repositori:
```sh
git clone https://github.com/wulanprr/UAP-ML.git
```
2. Navigasi:
```sh
cd UAP-ML
```
3. Buat dan aktifkan virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # Untuk Linux/MacOS
venv\Scripts\activate   # Untuk Windows
```
4. Install:
```sh
pip install -r requirements.txt
```
5. Run aplikasi menggunakan Streamlit:
```sh
python -m streamlit run uap.py
```

------------------------------------------
## Deskripsi Model
#### Model yang Digunakan
- Random Forest: Algoritma pembelajaran ensemble berbasis pohon keputusan (Decision Tree) yang bekerja dengan cara membangun beberapa pohon keputusan selama pelatihan dan menggabungkan hasilnya untuk menghasilkan prediksi akhir.
- Feedforward Neural Network (FNN): Tipe sederhana dari jaringan saraf tiruan (Artificial Neural Network). Informasi dalam FNN mengalir maju dari input ke output melalui lapisan tersembunyi tanpa adanya umpan balik (feedback).

---------------------------
## Hasil dan Analisi
### Random Forest
Model ini menunjukkan kinerja yang seimbang dalam mengklasifikasikan kedua kelas, dengan performa yang baik dalam hal precision, recall, dan f1-score, serta akurasi keseluruhan yang cukup tinggi. Model ini efektif dalam mengklasifikasikan data dengan mempertimbangkan kedua kelas secara merata.
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.93      | 0.74   | 0.82     | 11275   |
| 1     | 0.67      | 0.90   | 0.77     | 6634    |
|       |           |        |          |         |
| Accuracy |         |        | 0.80     | 17909   |
| Macro avg | 0.80    | 0.82   | 0.79     | 17909   |
| Weighted avg | 0.83 | 0.80   | 0.80     | 17909   |

### Feedforward Neural Network Evaluation
Model ini hampir sama dengan Random Forest karena menunjukkan kinerja yang seimbang dalam mengklasifikasikan kedua kelas, dengan performa yang baik dalam hal precision, recall, dan f1-score, serta akurasi keseluruhan yang cukup tinggi. Model ini efektif dalam mengklasifikasikan data dengan mempertimbangkan kedua kelas secara merata.
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.82      | 0.88   | 0.85     | 11275   |
| 1     | 0.76      | 0.68   | 0.72     | 6634    |
|       |           |        |          |         |
| Accuracy |         |        | 0.80     | 17909   |
| Macro avg | 0.79    | 0.78   | 0.78     | 17909   |
| Weighted avg | 0.80 | 0.80   | 0.80     | 17909   |

### Perbandingan Model
| Model                        | Akurasi (%) | Precision (%) | Recall (%) | F1-Score (%) |
|------------------------------|-------------|---------------|------------|--------------|
| Random Forest                | 80.00       | 83.00         | 80.00      | 80.00        |
| Feedforward Neural Network   | 80.00       | 80.00         | 80.00      | 78.00        |

----------------------
### Visualisasi Hasil Random Forest 
![cm_rf](https://github.com/user-attachments/assets/42048187-9eed-4b1f-ae57-30070c42cb13)

### Visualisasi Hasil Feedforward Neural Network
![cm_fnn](https://github.com/user-attachments/assets/d4c91c40-c0d2-4dd4-8214-02ccb51dd6f6)
Dengan grafik accuracy and loss berikut:
![accuracy and loss](https://github.com/user-attachments/assets/f9eb0198-0833-4ad5-80e0-0f8985e90020)


-------------------------------
## Local Web Deploy
### Hasil Prediksi 
![hasil prediksi](https://github.com/user-attachments/assets/99ee4d86-d191-440f-93bc-e15f85dc9a12)


------------
## Author
[Wulan Puspita Rahayu](https://github.com/wulanprr) - 20211070311199

