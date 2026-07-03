"""
predict_invoice.py
------------------
Inference file for invoice flagging classification.
Run ONLY after train.py has saved the model.

  python predict_invoice.py
"""

import joblib
import pandas as pd

MODEL_PATH  = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_model(model_path: str = MODEL_PATH):
    """Load trained invoice flagging model."""
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_flag_invoice(input_data: dict) -> pd.DataFrame:
    """
    Predict whether a vendor invoice should be flagged for manual review.

    Parameters
    ----------
    input_data : dict
        Keys must match training features used in train.py:
        invoice_quantity, invoice_dollars, Freight,
        days_po_to_invoice, total_item_quantity, total_item_dollars

    Returns
    -------
    pd.DataFrame with original data + Flag_Invoice column
        0 = normal invoice
        1 = flagged for manual review
    """
    model    = load_model()
    scaler   = joblib.load(SCALER_PATH)

    input_df = pd.DataFrame(input_data)
    scaled   = scaler.transform(input_df)

    input_df['Flag_Invoice'] = model.predict(scaled)
    input_df['Flag_Invoice'] = input_df['Flag_Invoice'].map({
        0: 'Normal',
        1: 'FLAGGED - Manual Review Needed'
    })
    return input_df


if __name__ == "__main__":
    # ── Example: check a few invoices ────────────────────────────────────────
    sample_data = {
        "invoice_quantity"   : [10,   500,  25],
        "invoice_dollars"    : [200,  95000, 800],
        "Freight"            : [15,   4500,  60],
        "days_po_to_invoice" : [3,    45,    7],
        "total_item_quantity": [10,   490,   25],
        "total_item_dollars" : [195,  50000, 790]
    }

    result = predict_flag_invoice(sample_data)
    print(result[['invoice_dollars', 'Freight', 'Flag_Invoice']])