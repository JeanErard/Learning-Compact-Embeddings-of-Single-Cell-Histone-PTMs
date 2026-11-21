from sklearn.model_selection import StratifiedKFold
import numpy as np

def add_folds(adata, n_splits=5):

    y = adata.obs["cell_type"].to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    adata.obs["fold"] = -1

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(adata.obsm["X_lsi"], y)
    ):
        adata.obs.iloc[
            test_idx,
            adata.obs.columns.get_loc("fold")
        ] = fold

    return adata