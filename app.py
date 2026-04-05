# app.py
# ============================================================
# Entry point aplikasi Prediksi Harga Emas XAUUSD
# Data upload user = TEST SET (model sudah dilatih sebelumnya)
# Jalankan dengan: streamlit run app.py
# ============================================================

import pandas as pd
import streamlit as st
import base64

from config import PAGE_CONFIG
from data_utils import clean_dataframe, feature_engineering
from model_utils import load_model_and_scaler, predict_next_day, run_prediction
from ui_components import (
    render_chart_tab,
    render_data_summary,
    render_metrics_tab,
    render_next_day_prediction,
    render_sidebar,
)

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)

# ── Load model & scaler ──────────────────────────────────────
model, scaler = load_model_and_scaler()

# ── Sidebar & upload file ────────────────────────────────────
uploaded_file = render_sidebar()

# ── Header utama ─────────────────────────────────────────────
st.title("Prediksi Harga Emas XAUUSD")
st.caption(
    "Stacked GRU  ·  Target: Log Return  ·  "
    "Model dilatih dengan 2.451 data historis  ·  "
    "Data upload digunakan sebagai Test Set"
)

if uploaded_file is None:
    st.info("📂 Upload file CSV XAUUSD di sidebar untuk memulai prediksi.")
    st.stop()

# ── Baca & proses data ───────────────────────────────────────
try:
    df_raw        = pd.read_csv(uploaded_file)
    df_raw        = clean_dataframe(df_raw)
    df_model_full = feature_engineering(df_raw)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

render_data_summary(df_model_full)

# ── Prediksi hari berikutnya ─────────────────────────────────
with st.spinner("Menghitung prediksi hari berikutnya..."):
    next_date, next_price, last_price, lr_pred = predict_next_day(
        model, scaler, df_model_full
    )

render_next_day_prediction(next_date, next_price, last_price, lr_pred)

# ── Evaluasi pada data upload (test set) ─────────────────────
st.divider()
st.subheader("Evaluasi model pada data yang diupload (Test Set)")
st.caption(
    "Model dievaluasi secara 1-step ahead: "
    "setiap prediksi menggunakan 30 hari sebelumnya sebagai input."
)

with st.spinner("Menjalankan evaluasi..."):
    results = run_prediction(model, scaler, df_model_full)

tab1, tab2 = st.tabs(["📐 Metrik Evaluasi", "📈 Grafik Prediksi"])

with tab1:
    render_metrics_tab(results)

with tab2:
    render_chart_tab(results)
