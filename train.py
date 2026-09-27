from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset() -> tuple[Any, Any, list[str]]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = (dataset.target == 0).astype("int8")
    return features, target, list(dataset.feature_names)


def build_searches(random_state: int, cv_folds: int) -> dict[str, GridSearchCV]:
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=random_state),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "rbf_svm": (
            SVC(kernel="rbf"),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", "auto"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [None, 8],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    searches = {}

    for name, (classifier, parameters) in candidates.items():
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("classifier", classifier)]
        )
        searches[name] = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

    return searches


def train_and_evaluate(
    output_dir: Path,
    test_size: float,
    random_state: int,
    cv_folds: int,
) -> dict[str, Any]:
    features, target, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    searches = build_searches(random_state, cv_folds)
    for search in searches.values():
        search.fit(X_train, y_train)

    best_name, best_search = max(
        searches.items(), key=lambda item: item[1].best_score_
    )
    best_model = best_search.best_estimator_
    predictions = best_model.predict(X_test)
    scores = (
        best_model.decision_function(X_test)
        if hasattr(best_model, "decision_function")
        else best_model.predict_proba(X_test)[:, 1]
    )
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["benign", "malignant"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    metrics_path = output_dir / "metrics.json"
    joblib.dump(best_model, model_path)

    summary = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn)",
        "positive_class": "malignant",
        "selected_model": best_name,
        "selection_metric": "mean cross-validation ROC-AUC",
        "candidate_cv_roc_auc": {
            name: float(search.best_score_) for name, search in searches.items()
        },
        "best_cv_roc_auc": float(best_search.best_score_),
        "test_roc_auc": float(roc_auc_score(y_test, scores)),
        "test_malignant_recall": float(report["malignant"]["recall"]),
        "test_benign_recall_specificity": float(report["benign"]["recall"]),
        "test_malignant_precision": float(report["malignant"]["precision"]),
        "confusion_matrix_labels": ["benign", "malignant"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "best_parameters": best_search.best_params_,
        "feature_names": feature_names,
        "model_file": model_path.name,
        "cv_folds": cv_folds,
        "test_size": test_size,
        "random_state": random_state,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Cross-validation ROC-AUC:")
    for name, search in sorted(
        searches.items(), key=lambda item: item[1].best_score_, reverse=True
    ):
        print(f"  {name}: {search.best_score_:.3f}")
    print(f"\nSelected model: {best_name}")
    print(f"Test ROC-AUC: {summary['test_roc_auc']:.3f}")
    print(f"Malignant recall: {summary['test_malignant_recall']:.3f}")
    print(f"Benign recall (specificity): {summary['test_benign_recall_specificity']:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; benign, malignant):")
    print(matrix)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print("Educational demonstration only; not for medical diagnosis.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare classifiers on the Wisconsin breast cancer dataset."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be greater than 0 and less than 1")
    if args.cv_folds < 2:
        raise SystemExit("--cv-folds must be at least 2")
    train_and_evaluate(
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()