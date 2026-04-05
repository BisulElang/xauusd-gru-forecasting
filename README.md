# 🥇 Prediksi Harga Emas XAUUSD — Stacked GRU

Aplikasi web untuk memprediksi harga emas XAUUSD menggunakan model **Stacked GRU (Gated Recurrent Unit)** berbasis deep learning.

## 🚀 Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📋 Fitur

- **Prediksi 1 hari ke depan** berdasarkan 30 hari terakhir data upload
- **Evaluasi model** (RMSE, MAE, MAPE) pada data yang diupload sebagai test set
- **Grafik interaktif** prediksi vs harga aktual
- **Download grafik** hasil prediksi (PNG)

---

## 🗂️ Struktur File

```
├── app.py                  # Entry point Streamlit
├── config.py               # Konfigurasi global
├── data_utils.py           # Preprocessing & feature engineering
├── model_utils.py          # Load model, prediksi, evaluasi
├── ui_components.py        # Komponen UI Streamlit
├── gru_model.keras         # Model GRU terlatih
├── scaler.pkl              # MinMaxScaler terlatih
├── requirements.txt        # Dependensi Python
├── packages.txt            # Dependensi sistem
└── .streamlit/
    └── config.toml         # Konfigurasi tema Streamlit
```

---

## 📊 Format CSV yang Didukung

Kolom wajib dalam file CSV:

| Kolom     | Keterangan              |
|-----------|-------------------------|
| `Date`    | Tanggal (MM/DD/YYYY)    |
| `Price`   | Harga penutupan         |
| `Open`    | Harga pembukaan         |
| `High`    | Harga tertinggi         |
| `Low`     | Harga terendah          |

> **Minimal 45 baris data** (±3 bulan) agar model dapat membentuk sekuens prediksi.

---

## ⚙️ Cara Menjalankan Lokal

```bash
# 1. Clone repo
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

---

## 🤖 Detail Model

| Parameter        | Nilai                          |
|------------------|-------------------------------|
| Arsitektur       | Stacked GRU                   |
| Target           | Log Return                    |
| Window size      | 30 hari                       |
| Fitur            | Log_Return, HL_Range, Rolling_Std_14 |
| Data training    | 2.451 data (Agt 2016 – Jan 2026) |

---

## 📌 Catatan

- Model **tidak dilatih ulang** saat deployment — data upload hanya digunakan sebagai **test set**.
- Prediksi bersifat indikatif dan **bukan saran investasi**.
