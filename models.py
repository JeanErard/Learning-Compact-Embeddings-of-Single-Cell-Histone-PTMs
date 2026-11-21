from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def tune_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5]
    }

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    return grid_search