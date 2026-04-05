# ui_components.py
# ============================================================
# Komponen antarmuka pengguna (Streamlit UI)
# Data upload = test set (model sudah dilatih sebelumnya)
# ============================================================

import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import FEATURES, WINDOW_SIZE


# ------------------------------------------------------------------ #
# Sidebar                                                               #
# ------------------------------------------------------------------ #

def render_sidebar() -> st.runtime.uploaded_file_manager.UploadedFile | None:
    """
    Menampilkan sidebar aplikasi dan mengembalikan file CSV yang diunggah.
    """
    with st.sidebar:
        st.title("Prediksi Harga Emas")
        st.caption("Model: Stacked GRU + Log Return")
        st.divider()

        uploaded_file = st.file_uploader(
            "Upload CSV data XAUUSD dengan minimal waktu 3 bulan (> 45 baris).",
            type=["csv"],
            help="Kolom wajib: Date, Price, Open, High, Low",
        )

        st.divider()
        st.markdown("**Model aktif:**")
        st.success("gru_model.keras")
        st.caption(f"Window size : {WINDOW_SIZE} hari")
        st.caption("Dilatih dengan : 2.451 data historis (Agustus 2016-Januari 2026)")
        st.caption(f"Fitur : {', '.join(FEATURES)}")
        st.divider()
        st.info(
            "Data yang Anda upload akan digunakan "
            "sebagai **test set** untuk mengevaluasi "
            "performa model pada data baru."
        )

    return uploaded_file


# ------------------------------------------------------------------ #
# Header & preview data                                                 #
# ------------------------------------------------------------------ #

def render_data_summary(df_model_full: pd.DataFrame) -> None:
    """
    Menampilkan ringkasan data yang berhasil dimuat dan preview 10 baris terakhir.
    """
    n_rows = len(df_model_full)
    n_seq  = max(0, n_rows - WINDOW_SIZE)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total baris data", f"{n_rows:,}")
    col2.metric("Sekuens yang akan diprediksi", f"{n_seq:,}")
    col3.metric(
        "Rentang data",
        f"{df_model_full['Date'].min().strftime('%d %b %Y')} "
        f"– {df_model_full['Date'].max().strftime('%d %b %Y')}",
    )

    with st.expander("Preview data (10 baris terakhir)"):
        st.dataframe(
            df_model_full[['Date', 'Price_actual'] + FEATURES].tail(10),
            use_container_width=True,
        )


# ------------------------------------------------------------------ #
# Prediksi hari berikutnya                                              #
# ------------------------------------------------------------------ #

def render_next_day_prediction(next_date, next_price: float,
                                last_price: float, lr_pred: float) -> None:
    """
    Menampilkan hasil prediksi harga 1 hari kerja berikutnya.
    """
    st.divider()
    st.subheader("🔮 Prediksi harga 1 hari berikutnya")

    change     = next_price - last_price
    change_pct = (change / last_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tanggal prediksi",    next_date.strftime('%d %b %Y'))
    col2.metric("Harga terakhir",      f"${last_price:,.2f}")
    col3.metric(
        "Prediksi besok",
        f"${next_price:,.2f}",
        delta=f"{change:+.2f} ({change_pct:+.2f}%)",
    )
    col4.metric("Log Return prediksi", f"{lr_pred:.6f}")

    st.caption(
        f"Basis prediksi: {WINDOW_SIZE} hari terakhir data upload  |  "
        f"Rumus: ${last_price:,.2f} × exp({lr_pred:.6f}) = ${next_price:,.2f}"
    )


# ------------------------------------------------------------------ #
# Tab 1: Metrik evaluasi (test set saja)                               #
# ------------------------------------------------------------------ #

def render_metrics_tab(results: dict) -> None:
    """
    Menampilkan metrik evaluasi (RMSE, MAE, MAPE) pada data upload (test set).
    """
    st.subheader("📐 Metrik evaluasi — data upload sebagai Test Set")

    st.info(
        f"Model dievaluasi pada **{results['n_sequences']} sekuens** "
        f"dari {results['n_rows_uploaded']} baris data yang diupload. "
        f"Baris pertama ({WINDOW_SIZE} baris) digunakan sebagai window awal."
    )

    te_rmse, te_mae, te_mape = results['test']

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"${te_rmse:.4f}", help="Root Mean Square Error (USD)")
    col2.metric("MAE",  f"${te_mae:.4f}",  help="Mean Absolute Error (USD)")
    col3.metric("MAPE", f"{te_mape:.4f}%", help="Mean Absolute Percentage Error")

    st.divider()

    st.dataframe(
        pd.DataFrame({
            'Metrik':  ['RMSE ($)', 'MAE ($)', 'MAPE (%)'],
            'Nilai':   [f"{te_rmse:.4f}", f"{te_mae:.4f}", f"{te_mape:.4f}"],
            'Satuan':  ['USD', 'USD', '%'],
            'Keterangan': [
                'Akar rata-rata kuadrat error',
                'Rata-rata absolut error',
                'Rata-rata persentase error absolut',
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("**Interpretasi MAPE:**")
    if te_mape < 10:
        st.success(f"MAPE {te_mape:.4f}% — **Sangat baik** (< 10%): Model bekerja sangat baik pada data baru.")
    elif te_mape < 20:
        st.warning(f"MAPE {te_mape:.4f}% — **Baik** (10–20%): Model cukup akurat pada data baru.")
    else:
        st.error(f"MAPE {te_mape:.4f}% — **Perlu ditingkatkan** (≥ 20%): Model kurang akurat pada data baru.")


# ------------------------------------------------------------------ #
# Tab 2: Grafik prediksi (test set)                                    #
# ------------------------------------------------------------------ #

def render_chart_tab(results: dict) -> None:
    """
    Menampilkan grafik prediksi vs aktual pada data upload (test set).
    """
    st.subheader("📈 Grafik prediksi vs harga aktual — Test Set")

    te_mape                        = results['test'][2]
    price_true, price_pred, dates  = (
        results['price_test_true'],
        results['price_test_pred'],
        results['test_dates'],
    )

    n = min(len(price_true), len(price_pred), len(dates))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates[:n], price_true[:n],
            label='Aktual',   color='steelblue', linewidth=1.8)
    ax.plot(dates[:n], price_pred[:n],
            label='Prediksi', color='orange',    linewidth=1.8, linestyle='--')

    ax.set_title(
        f"Prediksi Harga Emas XAUUSD — Test Set  |  MAPE: {te_mape:.4f}%",
        fontsize=12,
    )
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    fig.autofmt_xdate()
    plt.tight_layout()

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    st.download_button(
        "⬇️ Download grafik (PNG)",
        data=buf.getvalue(),
        file_name="prediksi_test_set.png",
        mime="image/png",
    )
