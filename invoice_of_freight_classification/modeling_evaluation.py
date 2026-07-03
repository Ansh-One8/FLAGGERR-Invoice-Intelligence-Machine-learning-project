"""
modeling_evaluation.py
----------------------
Trains Random Forest with GridSearchCV and evaluates it.
Knows nothing about data loading or file paths.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
    classification_report
)


# ── 1. Train ──────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> GridSearchCV:
    """
    Train a RandomForestClassifier with GridSearchCV.
    Optimises for F1 score (better than accuracy for imbalanced labels).

    Returns the fitted GridSearchCV object — call .best_estimator_
    to get the actual best model out of it.
    """
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1           # use all CPU cores
    )

    param_grid = {
        'n_estimators'     : [100, 200, 300],
        'max_depth'        : [None, 4, 5, 6],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf' : [1, 2, 5],
        'criterion'        : ['gini', 'entropy']
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator  = rf,
        param_grid = param_grid,
        scoring    = scorer,
        cv         = 5,
        verbose    = 2,
        n_jobs     = -1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n[grid search] Best params : {grid_search.best_params_}")
    print(f"[grid search] Best F1     : {grid_search.best_score_:.4f}")

    return grid_search


# ── 2. Evaluate ───────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, model_name: str):
    """
    Print full classification metrics for a trained model.

    Metrics explained:
      Accuracy  – % of total predictions correct
      Precision – of all flagged invoices, how many were actually risky
      Recall    – of all actually risky invoices, how many did we catch
      F1        – balance between precision and recall (most important here)
      Confusion Matrix – shows exactly where the model is making mistakes
    """
    preds = model.predict(X_test)

    print(f"\n{'─'*45}")
    print(f"  {model_name}")
    print(f"{'─'*45}")
    print(f"  Accuracy  : {accuracy_score(y_test, preds)*100:.2f}%")
    print(f"  Precision : {precision_score(y_test, preds)*100:.2f}%")
    print(f"  Recall    : {recall_score(y_test, preds)*100:.2f}%")
    print(f"  F1 Score  : {f1_score(y_test, preds)*100:.2f}%")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
    print(f"\n  Classification Report:\n{classification_report(y_test, preds)}")