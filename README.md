# 🚢 FLAGGERR — Invoice Intelligence Platform {first ML project}

**FLAGGERR** is a machine-learning-powered Streamlit app that helps finance and logistics teams catch bad invoices before they cause problems. It does two things:

1. **Predicts freight cost** from an invoice dollar amount using a Linear Regression model.
2. **Flags risky invoices** — abnormal cost discrepancies or delivery delays — using a tuned Random Forest classifier.

Built with a clean, Apple-inspired dark UI, smooth animations, and instant on-screen predictions — no spreadsheets, no manual cross-checking.

---

## ✨ Features

- 🧮 **Freight Cost Predictor** — enter an invoice amount, get an instant predicted freight cost with the freight-to-invoice ratio.
- 🚨 **Invoice Risk Detector** — enter invoice quantity, dollars, freight, PO-to-invoice days, and item totals to get a Normal / Flagged verdict with confidence scores.
- 📊 Model performance surfaced directly in the UI (R², MAE, RMSE, Accuracy, Precision, Recall).
- 🎨 Apple-style dark interface with gradient text, glass cards, and smooth micro-interactions.
- ⚡ Fast, cached model loading via `joblib` + `st.cache_resource`.

---

## 🧠 Models

| Task | Model | Key Metrics |
|---|---|---|
| Freight Cost Prediction | Linear Regression | R² 96.99% · MAE 24.11 · RMSE 124.72 |
| Invoice Risk Classification | Random Forest (GridSearchCV-tuned) | Accuracy 89% · Precision 96% · Recall 71% |

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — UI framework
- **scikit-learn** — model training & inference
- **pandas / numpy** — data processing
- **joblib** — model serialization
- **SQLite** — data storage

---

## 📂 Project Structure

```
FLAGGERR/
├── app.py                                  # Main Streamlit application
├── freight_prediction_project/
│   ├── main.py                             # Trains the freight cost regression model
│   └── saved_models/
│       ├── predict_freight_model.pkl
│       └── scaler.pkl
├── invoice_of_freight_classification/
│   ├── train.py                            # Trains the invoice risk classifier
│   └── models/
│       ├── predict_flag_invoice.pkl
│       └── scaler.pkl
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/flaggerr.git
cd flaggerr
```

### 2. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### 3. Train the models (first run only)

```bash
python freight_prediction_project/main.py
python invoice_of_freight_classification/train.py
```

This generates the `saved_models/` and `models/` folders with the `.pkl` files the app needs.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🖥️ Usage

**Freight Cost Predictor**
1. Go to the *Freight Cost Predictor* tab in the sidebar.
2. Enter the invoice amount in USD.
3. Click **Calculate Freight Cost →** to see the predicted freight cost and freight ratio.

**Invoice Risk Detector**
1. Go to the *Invoice Risk Detector* tab.
2. Fill in invoice quantity, invoice dollars, freight cost, days from PO to invoice, total item quantity, and total item dollars.
3. Click **Analyse Invoice Risk →** to get a Normal/Flagged verdict, confidence breakdown, and dollar discrepancy.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 👤 Author

**Ansh Mishra** and Special thanks to ayushi mishra for this she was the inspiration behind this and without her i wont be able to this 
highly suggest to go and checkout her youtube channel.

Designed and built FLAGGERR — from data pipeline to UI. 
