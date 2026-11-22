from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_models(random_state=42):
    """Model catalogue: {name: (estimator, hyperparameter grid)}.

    Scale-sensitive models (logreg, kNN, SVM, MLP) have a StandardScaler in their
    Pipeline: it is fitted on the training set of each fold only.
    To add a model, just add an entry here.
    """
    return {
        # Linear probe: what the embedding separates linearly
        "logreg": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),

        # kNN: do a cell's neighbours share its type?
        "knn": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {
                "clf__n_neighbors": [5, 15, 30],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "cosine"],
            },
        ),

        # SVM with RBF kernel
        "svm": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced")),
            ]),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01],
            },
        ),

        # Small neural network (no class_weight available in sklearn)
        "mlp": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(early_stopping=True, max_iter=500,
                                      random_state=random_state)),
            ]),
            {
                "clf__hidden_layer_sizes": [(128,), (128, 64)],
                "clf__alpha": [1e-4, 1e-3, 1e-2],
            },
        ),

        # Lightweight random forest: n_estimators and max_features fixed, 4 combinations
        # instead of 54 (RF is not very sensitive to these parameters)
        "rf": (
            RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   class_weight="balanced", random_state=random_state,
                                   n_jobs=1),
            {
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 3],
            },
        ),
    }


MODEL_NAMES = list(get_models().keys())


def tune_model(name, X_train, y_train, inner_splits=3, random_state=42, n_jobs=-1):
    """GridSearchCV with inner CV on the training set only, then refit on the whole training set."""
    models = get_models(random_state)
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Available: {list(models)}")
    estimator, param_grid = models[name]

    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                               random_state=random_state)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=n_jobs,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def tune_random_forest(X_train, y_train):
    """Kept for backward compatibility with the previous code."""
    return tune_model("rf", X_train, y_train)
