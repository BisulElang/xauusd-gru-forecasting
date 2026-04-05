# config.py
# ============================================================
# Konfigurasi global aplikasi Prediksi Harga Emas (GRU)
# ============================================================

# --- Path file model & scaler ---
MODEL_PATH  = "gru_model.keras"
SCALER_PATH = "scaler.pkl"

# --- Parameter model ---
WINDOW_SIZE = 30
FEATURES    = ['Log_Return', 'HL_Range', 'Rolling_Std_14']

# --- Rasio pembagian dataset ---
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = sisa (0.15) — dihitung otomatis dari TRAIN + VAL

# --- Konfigurasi halaman Streamlit ---
PAGE_CONFIG = {
    "page_title": "Prediksi Harga Emas — GRU",
    "layout":     "wide",
}
