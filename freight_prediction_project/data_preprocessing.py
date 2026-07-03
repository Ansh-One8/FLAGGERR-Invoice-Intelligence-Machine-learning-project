"""
data_preprocessing.py
Handles everything BEFORE model training:
  - Loading data from SQLite
  - Cleaning (dropping nulls, bad columns)
  - Feature + target selection
  - Scaling
  - Train/test split
"""

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """Load vendor_invoice table from a SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
    conn.close()
    return df




def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful for modelling.
    Add engineered features (freight_per_unit).
    """
    # Drop the Approval column – it had all nulls
    df = df.drop(columns=["Approval"], errors="ignore")

    # Feature engineering: cost efficiency signal
    df["freight_per_unit"] = df["Freight"] / df["Quantity"]

    return df


#Prepare features & target 

def prepare_features(df: pd.DataFrame):
    """
    Select X (features) and y (target).

    We use only 'Dollars' as input feature here because:
      - Quantity & Dollars are both correlated with Freight (see heatmap)
      - Dollars showed the strongest linear relationship in scatter plots
    You can extend X to [['Dollars', 'Quantity']] to experiment.
    """
    X = df[["Dollars"]]
    y = df["Freight"]          
    return X, y


#  Scale 

def scale_features(X_train, X_test):
    """
    Fit StandardScaler on TRAIN only, then transform both sets.

    Why fit on train only?
      If you fit on the whole dataset first you 'leak' test-set statistics
      into training – the model gets an unfair peek at future data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std HERE
    X_test_scaled  = scaler.transform(X_test)         # apply same scale HERE
    return X_train_scaled, X_test_scaled, scaler


# Spliting

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def run_preprocessing(db_path: str):
    """
    Single function that calls every step above in order.
    main.py just needs to call this one function.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    df          = load_vendor_invoice_data(db_path)
    df          = clean_data(df)
    X, y        = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler      = scale_features(X_train, X_test)

    print(f"[preprocessing] train size={len(X_train_s)}  test size={len(X_test_s)}")
    return X_train_s, X_test_s, y_train, y_test, scaler