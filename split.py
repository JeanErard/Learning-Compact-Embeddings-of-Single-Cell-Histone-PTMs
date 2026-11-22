import numpy as np
from sklearn.model_selection import StratifiedKFold

from labels import group_cell_types


def add_folds(adata, n_splits=5, label_key="cell_type_grouped", random_state=42):
    """Add adata.obs["fold"] (0..n_splits-1), stratified on label_key.

    Folds only depend on the labels, so they are identical for every embedding
    (X_lsi, X_ae, ...) and every model.
    If label_key is "cell_type_grouped" and does not exist yet, classes are
    grouped automatically (see labels.py).
    """
    if label_key not in adata.obs:
        if label_key == "cell_type_grouped":
            group_cell_types(adata)
        else:
            raise KeyError(f"'{label_key}' not found in adata.obs")

    y = adata.obs[label_key].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = np.full(adata.n_obs, -1, dtype=int)
    for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[test_idx] = fold

    adata.obs["fold"] = folds
    return adata
