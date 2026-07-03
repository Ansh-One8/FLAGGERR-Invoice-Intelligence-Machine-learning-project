"""
main.py
-------
The ONLY file you need to run.
It just calls functions from the other two files in order.

Run it:   python main.py
"""

import matplotlib.pyplot as plt
import pandas as pd

# Import our own modules (the two files we wrote)
from data_preprocessing import run_preprocessing
from model_training import (
    get_models,
    train_all_models,
    evaluate_all_models,
    get_best_model,
    save_model,
)

# Config 
# Change this path to wherever your .db file lives
DB_PATH = r"C:\Users\Ansh Mishra\Desktop\PythoncourseML\datasets\inventory.db"


# Pipeline 

def main():

    # STEP 1 – Load, clean, split, scale  (all inside data_preprocessing.py)
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(DB_PATH)

    # STEP 2 – Get model definitions  (model_training.py)
    models = get_models()

    # STEP 3 – Train every model
    trained_models = train_all_models(models, X_train, y_train)

    # STEP 4 – Evaluate every model and print a comparison
    results = evaluate_all_models(trained_models, X_test, y_test)

    # STEP 5 – Pick the winner
    best_name, best_model = get_best_model(trained_models, results)

    # STEP 6 – Save best model + scaler to disk (so predict.py can use them)
    save_model(best_model, scaler)

    print("\n[plot] Showing predictions vs actual for best model…")
    preds = best_model.predict(X_test)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test,  label="Actual Freight",     alpha=0.5)
    plt.scatter(X_test, preds,   label=f"{best_name} preds", alpha=0.5, color="red")
    plt.xlabel("Dollars (scaled)")
    plt.ylabel("Freight")
    plt.title(f"Best model: {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n[summary] Model comparison:")
    print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))


if __name__ == "__main__":
    main()