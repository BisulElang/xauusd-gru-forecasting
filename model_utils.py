# model_utils.py
# ============================================================
# Utilitas model: load, prediksi, evaluasi
# Data yang diupload user digunakan sebagai TEST SET saja.
# Model GRU sudah dilatih sebelumnya dengan data historis.
# ============================================================

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    WINDOW_SIZE,
)
from data_utils import create_sequences


# ------------------------------------------------------------------ #
# Load model & scaler                                                  #
# ------------------------------------------------------------------ #

@st.cache_resource
def load_model_and_scaler():
    """
    Memuat model GRU dan scaler dari disk.

    Model dan scaler hanya dimuat sekali dan di-cache oleh Streamlit
    untuk menghindari pemuatan ulang di setiap interaksi pengguna.

    Returns
    -------
    model : tf.keras.Model
    scaler : sklearn scaler (MinMaxScaler / StandardScaler)

    Raises
    ------
    st.stop() jika salah satu file tidak ditemukan.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"File '{MODEL_PATH}' tidak ditemukan. "
            "Pastikan sudah menjalankan notebook terlebih dahulu."
        )
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error(
            f"File '{SCALER_PATH}' tidak ditemukan. "
            "Pastikan sudah menjalankan notebook terlebih dahulu."
        )
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


# ------------------------------------------------------------------ #
# Fungsi transformasi                                                   #
# ------------------------------------------------------------------ #

def inverse_log_return(scaler, scaled_y: np.ndarray) -> np.ndarray:
    """
    Mengembalikan nilai Log Return dari skala normalisasi ke skala asli.

    Scaler di-fit pada 3 kolom (FEATURES), sehingga inverse_transform
    membutuhkan input dengan 3 kolom. Kolom ke-0 diisi scaled_y;
    kolom sisanya diisi 0 sebagai placeholder.
    """
    placeholder = np.zeros((len(scaled_y), len(FEATURES)))
    placeholder[:, 0] = scaled_y.flatten()
    return scaler.inverse_transform(placeholder)[:, 0]


def reconstruct_price(log_return_pred: np.ndarray,
                      price_prev: np.ndarray) -> np.ndarray:
    """
    Merekonstruksi harga dari prediksi Log Return.
    Rumus: P_t = P_{t-1} × exp(LogReturn_t)
    """
    return price_prev * np.exp(log_return_pred)


# ------------------------------------------------------------------ #
# Evaluasi                                                              #
# ------------------------------------------------------------------ #

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Menghitung metrik evaluasi regresi: RMSE, MAE, MAPE.

    Returns
    -------
    (rmse, mae, mape) : tuple[float, float, float]
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape


# ------------------------------------------------------------------ #
# Evaluasi pada data upload (full sebagai test set)                    #
# ------------------------------------------------------------------ #

def run_prediction(model, scaler, df_model_full: pd.DataFrame) -> dict:
    """
    Menjalankan prediksi 1-step ahead pada seluruh data yang diupload user.

    Data upload diperlakukan sepenuhnya sebagai TEST SET karena model
    GRU sudah dilatih sebelumnya dengan data historis terpisah (2451 data).

    Proses:
    1. Validasi jumlah baris mencukupi untuk membentuk minimal 1 sekuens.
    2. Scale fitur menggunakan scaler yang sudah di-fit saat training.
    3. Bentuk sekuens (X) dengan WINDOW_SIZE.
    4. Prediksi Log Return → inverse transform → rekonstruksi harga.
    5. Hitung metrik evaluasi.

    Parameters
    ----------
    model  : tf.keras.Model
    scaler : sklearn scaler
    df_model_full : pd.DataFrame
        Output dari feature_engineering() — seluruhnya adalah data test.

    Returns
    -------
    dict berisi metrik, array harga (aktual & prediksi), dan tanggal.
    """
    n = len(df_model_full)

    # ── Validasi minimum data ────────────────────────────────
    MIN_ROWS = WINDOW_SIZE + 1
    if n < MIN_ROWS:
        st.error(
            f"Data terlalu sedikit: "
            f"Dibutuhkan minimal 44 baris + 1 header"
            f"(masukan data dengan minimal rentang waktu harian selama 3 bulan) agar model dapat membentuk sekuens prediksi."
        )
        st.stop()

    price_actual = df_model_full['Price_actual'].values

    # ── Scale & bentuk sekuens ───────────────────────────────
    data_scaled = scaler.transform(df_model_full[FEATURES])
    X, _        = create_sequences(data_scaled, WINDOW_SIZE)

    if len(X) == 0:
        st.error(
            f"Tidak ada sekuens yang terbentuk. "
            f"Pastikan data memiliki lebih dari {WINDOW_SIZE} baris."
        )
        st.stop()

    # ── Prediksi ─────────────────────────────────────────────
    y_pred_scaled = model.predict(X, verbose=0)
    lr_pred       = inverse_log_return(scaler, y_pred_scaled.flatten())

    # ── Rekonstruksi harga ───────────────────────────────────
    # price_actual[WINDOW_SIZE:]    = harga aktual hari yang diprediksi
    # price_actual[WINDOW_SIZE-1:-1]= harga hari sebelumnya (basis rekonstruksi)
    price_true = price_actual[WINDOW_SIZE:]
    price_prev = price_actual[WINDOW_SIZE - 1 : len(price_actual) - 1]

    n_valid    = min(len(price_true), len(price_prev), len(lr_pred))
    price_true = price_true[:n_valid]
    price_prev = price_prev[:n_valid]
    lr_pred    = lr_pred[:n_valid]

    price_pred = reconstruct_price(lr_pred, price_prev)

    # ── Tanggal ──────────────────────────────────────────────
    dates      = pd.to_datetime(df_model_full['Date'].values)
    test_dates = dates[WINDOW_SIZE:][:n_valid]

    return {
        'test':            compute_metrics(price_true, price_pred),
        'price_test_true': price_true,
        'price_test_pred': price_pred,
        'test_dates':      test_dates,
        'n_sequences':     n_valid,
        'n_rows_uploaded': n,
    }


# ------------------------------------------------------------------ #
# Prediksi 1 hari berikutnya                                           #
# ------------------------------------------------------------------ #

def predict_next_day(model, scaler,
                     df_model_full: pd.DataFrame) -> tuple:
    """
    Memprediksi harga emas 1 hari kerja setelah tanggal terakhir data.

    Langkah:
    1. Ambil WINDOW_SIZE baris terakhir sebagai input model.
    2. Scale → reshape → prediksi Log Return.
    3. Inverse transform Log Return.
    4. Rekonstruksi harga: P_next = P_last × exp(lr_pred).
    5. Hitung tanggal prediksi (hari kerja berikutnya).
    """
    last_window_raw    = df_model_full[FEATURES].values[-WINDOW_SIZE:]
    last_window_scaled = scaler.transform(last_window_raw)
    X_next             = last_window_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))

    lr_pred_scaled = model.predict(X_next, verbose=0)[0, 0]

    placeholder        = np.zeros((1, len(FEATURES)))
    placeholder[0, 0]  = lr_pred_scaled
    lr_pred            = scaler.inverse_transform(placeholder)[0, 0]

    last_price = df_model_full['Price_actual'].values[-1]
    next_price = last_price * np.exp(lr_pred)

    last_date  = pd.to_datetime(df_model_full['Date'].values[-1])
    next_date  = last_date + pd.offsets.BDay(1)

    return next_date, next_price, last_price, lr_pred
