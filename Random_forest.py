from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from preprocessing import preprocessing
import pandas as pd


def random_forest(adata_train, adata_val, param_grid):

    results = []

    y_train = adata_train.obs["cell_type"].to_numpy()
    y_val = adata_val.obs["cell_type"].to_numpy()

    # Loop over preprocessing parameters
    for factor in param_grid["factor"]:
        for threshold in param_grid["threshold"]:

            X_train, X_val = preprocessing(
                adata_train, adata_val,
                factor=factor,
                threshold=threshold
            )

            # Loop over Random Forest parameters
            rf_grid = {
                "n_estimators": param_grid["n_estimators"],
                "max_depth": param_grid["max_depth"],
                "min_samples_leaf": param_grid["min_samples_leaf"]
            }

            for params in ParameterGrid(rf_grid):

                rf = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=42,
                    n_jobs=-1
                )

                rf.fit(X_train, y_train)

                y_pred = rf.predict(X_val)

                score = balanced_accuracy_score(y_val, y_pred)

                results.append({
                    "factor": factor,
                    "threshold": threshold,
                    **params,
                    "balanced_accuracy": score
                })

    results_df = pd.DataFrame(results)

    best_params = results_df.loc[
        results_df["balanced_accuracy"].idxmax()
    ]

    print(results_df.sort_values(
        "balanced_accuracy", ascending=False
    ).head())

    return results_df, best_params