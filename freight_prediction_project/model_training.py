"""
model_training.py
Defines, trains, and compares multiple regression models.
Completely separate from data loading or evaluation logic.
"""

import math
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#  Build models 

def get_models() -> dict:
    """
    Return a dict of {name: untrained_model}.
    To try a new model just add it here – nothing else needs to change.
    """
    return {
        "LinearRegression":      LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
    }


#  Train all models 

def train_all_models(models: dict, X_train, y_train) -> dict:
    """
    Fit every model in the dict on the training data.
    Returns the same dict with trained (fitted) models.
    """
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"[training] ✓ {name} trained")
    return models


#  Evaluate one model 

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Predict on test set and print + return performance metrics.

    Metrics explained (simple version):
      MAE  – average error in same units as Freight (lower = better)
      RMSE – same but punishes big mistakes harder (lower = better)
      R2  – % of variance explained by the model (higher = better, max 100%)
    """
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)
    r2   = r2_score(y_test, preds) * 100

    print(f"\n{'─'*40}")
    print(f"  {model_name}")
    print(f"{'─'*40}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R2  : {r2:.2f}%")

    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# Evaluate all models

def evaluate_all_models(trained_models: dict, X_test, y_test) -> list:
    """
    Run evaluate_model() for every trained model.
    Returns a list of result dicts – useful for building a comparison table.
    """
    results = []
    for name, model in trained_models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
    return results



def get_best_model(trained_models: dict, results: list) -> tuple:
    """
    Return (best_model_name, best_model_object) based on lowest RMSE.
    """
    best = min(results, key=lambda r: r["RMSE"])
    best_name  = best["model"]
    best_model = trained_models[best_name]
    print(f"\n[selection] Best model → {best_name}  (RMSE={best['RMSE']:.2f})")
    return best_name, best_model



def save_model(model, scaler, model_dir: str = "saved_models"):
    """
    Save the best model AND the scaler to disk using joblib.

    Why save the scaler too?
      When a new invoice comes in, you need to scale its 'Dollars'
      value the SAME way you scaled training data. Without the saved
      scaler you'd get wrong predictions.

    Creates:
      saved_models/predict_freight_model.pkl  ← the trained model
      saved_models/scaler.pkl                 ← the fitted scaler
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)     

    model_path  = model_dir / "predict_freight_model.pkl"
    scaler_path = model_dir / "scaler.pkl"

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n[saved] Model  → {model_path}")
    print(f"[saved] Scaler → {scaler_path}")
    return model_path, scaler_path



def load_model(model_dir: str = "saved_models"):
    """
    Load the saved model and scaler back from disk.
    Used by predict.py — no retraining needed.
    """
    model_dir   = Path(model_dir)
    model  = joblib.load(model_dir / "predict_freight_model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    print(f"[loaded] Model and scaler from {model_dir}/")
    return model, scaler