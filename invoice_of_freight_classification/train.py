"""
train.py
--------
Run this file to train and save the model.

  python train.py

Does NOT contain any logic — just calls functions
from data_preprocessing.py and modeling_evaluation.py in order.
"""

import joblib
from pathlib import Path

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features,
    FEATURES,
    TARGET
)
from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)


def main():

    # STEP 1 – Load data from SQLite + merge purchases + compute date features
    df = load_invoice_data()

    # STEP 2 – Create the flag_invoice label column using business rules
    df = apply_labels(df)

    # STEP 3 – Split into train / test
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # STEP 4 – Scale (saves scaler.pkl to models/ folder automatically)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test,
        scaler_path='models/scaler.pkl'
    )

    # STEP 5 – Train Random Forest with GridSearchCV (optimised for F1)
    grid_search = train_random_forest(X_train_scaled, y_train)

    # STEP 6 – Evaluate the best model found by grid search
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # STEP 7 – Save best model to disk
    Path("models").mkdir(exist_ok=True)
    model_path = "models/predict_flag_invoice.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n[saved] Best model → {model_path}")


if __name__ == "__main__":
    main()