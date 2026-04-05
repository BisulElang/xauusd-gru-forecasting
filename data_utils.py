# data_utils.py
# ============================================================
# Utilitas pemrosesan data: cleaning & feature engineering
# ============================================================

import numpy as np
import pandas as pd

from config import FEATURES


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan DataFrame mentah hasil upload CSV.

    Langkah:
    - Parse kolom 'Date' menjadi datetime.
    - Urutkan berdasarkan tanggal (ascending).
    - Hapus pemisah ribuan pada kolom harga (Price, Open, High, Low).
    - Bersihkan simbol '%' pada kolom 'Change %' jika ada.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mentah hasil pd.read_csv().

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah dibersihkan.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    if 'Change %' in df.columns:
        df['Change %'] = (
            df['Change %']
            .astype(str)
            .str.replace('%', '')
        )
        df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur-fitur teknikal dari data harga.

    Fitur yang dibuat:
    - Log_Return      : logaritma natural dari rasio harga hari ini / hari sebelumnya.
    - HL_Range        : selisih harga High dan Low.
    - Rolling_Std_14  : standar deviasi bergulir 14 hari dari harga penutupan.
    - Price_actual    : salinan kolom 'Price' sebagai target aktual.

    Baris yang mengandung NaN (akibat shift/rolling) otomatis dihapus.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame hasil clean_dataframe().

    Returns
    -------
    pd.DataFrame
        DataFrame berisi kolom FEATURES + ['Price_actual', 'Date'].
    """
    df = df.copy()
    df['Log_Return']     = np.log(df['Price'] / df['Price'].shift(1))
    df['HL_Range']       = df['High'] - df['Low']
    df['Rolling_Std_14'] = df['Price'].rolling(window=14).std()
    df['Price_actual']   = df['Price']

    df_model = (
        df[FEATURES + ['Price_actual', 'Date']]
        .dropna()
        .reset_index(drop=True)
    )
    return df_model


def create_sequences(data: np.ndarray, window_size: int):
    """
    Membentuk pasangan (X, y) untuk input model sekuensial (GRU/LSTM).

    Setiap sampel X berisi 'window_size' langkah waktu, sedangkan y adalah
    nilai pada langkah berikutnya (kolom ke-0, yaitu Log_Return).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data yang sudah di-scale.
    window_size : int
        Panjang jendela waktu (time-step).

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size, n_features)
    y : np.ndarray, shape (n_windows,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
