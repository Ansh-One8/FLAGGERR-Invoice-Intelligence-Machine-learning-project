"""
data_preprocessing.py
---------------------
Loads, merges, labels, splits and scales data.
Knows nothing about models.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH  = r"C:\Users\Ansh Mishra\Desktop\PythoncourseML\datasets\inventory.db"

FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'days_po_to_invoice',
    'total_item_quantity',
    'total_item_dollars'
]
TARGET = 'flag_invoice'


# ── 1. Load & merge ───────────────────────────────────────────────────────────

def load_invoice_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load vendor_invoice and purchases from SQLite,
    compute date features, aggregate purchases by PONumber,
    then LEFT JOIN both into one final dataframe.
    """
    conn = sqlite3.connect(db_path)

    purchases_df = pd.read_sql_query("SELECT * FROM purchases", conn)
    vendor_invoice_df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
    conn.close()

    # ── purchases: compute receiving delay then aggregate per PO ──────────────
    purchases_df['receiving_delay'] = (
        pd.to_datetime(purchases_df['ReceivingDate']) -
        pd.to_datetime(purchases_df['PODate'])
    ).dt.days

    purchase_avg_df = purchases_df.groupby('PONumber').agg(
        total_brands        = ('Brand',           'nunique'),
        total_item_quantity = ('Quantity',         'sum'),
        total_item_dollars  = ('Dollars',          'sum'),
        avg_receiving_delay = ('receiving_delay',  'mean')
    ).reset_index()

    # ── vendor_invoice: compute date diff features ────────────────────────────
    vendor_invoice_df['days_po_to_invoice'] = (
        pd.to_datetime(vendor_invoice_df['InvoiceDate']) -
        pd.to_datetime(vendor_invoice_df['PODate'])
    ).dt.days

    vendor_invoice_df['days_to_pay'] = (
        pd.to_datetime(vendor_invoice_df['PayDate']) -
        pd.to_datetime(vendor_invoice_df['InvoiceDate'])
    ).dt.days

    vi_df = vendor_invoice_df[[
        'PONumber', 'Quantity', 'Dollars', 'Freight',
        'days_po_to_invoice', 'days_to_pay'
    ]].rename(columns={
        'Quantity': 'invoice_quantity',
        'Dollars' : 'invoice_dollars'
    })

    # ── LEFT JOIN ─────────────────────────────────────────────────────────────
    final_df = vi_df.merge(purchase_avg_df, on='PONumber', how='left')

    print(f"[load] final_df shape: {final_df.shape}")
    return final_df


# ── 2. Label ──────────────────────────────────────────────────────────────────

def create_invoice_risk_label(row) -> int:
    """
    Business rule to flag an invoice:
      1 = flagged (needs manual review)
      0 = normal
    """
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1
    if row["avg_receiving_delay"] > 10:
        return 1
    return 0


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the risk label to every row."""
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    print(f"[labels] distribution:\n{df['flag_invoice'].value_counts()}")
    return df


# ── 3. Split ──────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, features: list, target: str):
    """Select features + target and do train/test split."""
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ── 4. Scale ──────────────────────────────────────────────────────────────────

def scale_features(X_train, X_test, scaler_path: str = "models/scaler.pkl"):
    """
    Fit StandardScaler on train only, transform both.
    Saves the scaler to disk so predict.py can reuse it.
    """
    Path(scaler_path).parent.mkdir(exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"[scale] Scaler saved → {scaler_path}")

    return X_train_scaled, X_test_scaled