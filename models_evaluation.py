import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from models import MODEL_NAMES, tune_model


def evaluate_model(adata, model_name, emb_key="X_lsi", label_key="cell_type_grouped",
                   store_predictions=True, verbose=True):
    """Nested CV of one model on one embedding.

    Outer loop: folds from adata.obs["fold"] (see split.add_folds).
    Inner loop: GridSearchCV on the fold's training set (models.tune_model).
    Predictions on each test fold are stored in
    adata.obs[f"pred_{model_name}_{emb_key}"] (confusion matrix, per-class F1...).
    """
    if "fold" not in adata.obs:
        raise KeyError("adata.obs['fold'] not found: call split.add_folds first")

    X = np.asarray(adata.obsm[emb_key])
    y = adata.obs[label_key].astype(str).to_numpy()
    folds = adata.obs["fold"].to_numpy()

    y_pred_all = np.empty(len(y), dtype=object)
    results = []

    for fold in np.unique(folds):
        train_idx = folds != fold
        test_idx = folds == fold
        t0 = time.time()

        # Hyperparameter tuning on the training set only
        grid_search = tune_model(model_name, X[train_idx], y[train_idx])

        # Evaluation on the outer test fold
        y_pred = grid_search.best_estimator_.predict(X[test_idx])
        y_pred_all[test_idx] = y_pred

        results.append({
            "model": model_name,
            "embedding": emb_key,
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], y_pred),
            "macro_f1": f1_score(y[test_idx], y_pred, average="macro", zero_division=0),
            "best_params": grid_search.best_params_,
            "inner_cv_score": grid_search.best_score_,
        })
        if verbose:
            r = results[-1]
            print(f"[{model_name} | {emb_key}] fold {fold} : "
                  f"bal_acc={r['balanced_accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}  "
                  f"({time.time() - t0:.0f}s)")

    if store_predictions:
        adata.obs[f"pred_{model_name}_{emb_key}"] = pd.Categorical(y_pred_all)

    return pd.DataFrame(results)


def evaluate_random_forest(adata, **kwargs):
    """Kept for backward compatibility with the previous notebook."""
    return evaluate_model(adata, "rf", **kwargs)


def evaluate_all(adata, model_names=None, emb_key="X_lsi", label_key="cell_type_grouped",
                 verbose=True):
    """Evaluate several models on one embedding.

    Returns (per-fold scores, mean ± std summary per model).
    """
    model_names = model_names or MODEL_NAMES
    per_fold = pd.concat(
        [evaluate_model(adata, name, emb_key=emb_key, label_key=label_key, verbose=verbose)
         for name in model_names],
        ignore_index=True,
    )
    summary = (
        per_fold.groupby(["embedding", "model"], sort=False)[["balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(3)
    )
    return per_fold, summary
