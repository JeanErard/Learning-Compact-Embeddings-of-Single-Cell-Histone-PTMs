import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score, f1_score
from models import tune_random_forest


def evaluate_random_forest(adata):

    X = adata.obsm["X_lsi"]
    y = adata.obs["cell_type"].to_numpy()
    folds = adata.obs["fold"].to_numpy()

    results = []

    for fold in np.unique(folds):

        # Outer split
        train_idx = folds != fold
        test_idx = folds == fold

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Hyperparameter tuning on training data only
        grid_search = tune_random_forest(X_train, y_train)

        # Best model, refitted on the entire outer training set
        best_model = grid_search.best_estimator_

        # Evaluate on outer test fold
        y_pred = best_model.predict(X_test)

        results.append({
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "best_params": grid_search.best_params_,
            "inner_cv_score": grid_search.best_score_
        })

    return pd.DataFrame(results)