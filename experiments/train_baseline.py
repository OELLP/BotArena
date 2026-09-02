from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.misbot_io import DEFAULT_MISBOT_ROOT, MisBotDataset, PROJECT_ROOT, extract_user_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first explainable MisBot account baseline")
    parser.add_argument("--root", type=Path, default=DEFAULT_MISBOT_ROOT)
    parser.add_argument("--full", action="store_true", help="Use all 99,874 manually annotated users")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    dataset = MisBotDataset(args.root)
    dataset.require_complete()
    users = list(dataset.iter_users(sampled=not args.full, limit=args.limit))
    labelled = [user for user in users if user.label in (0, 1)]
    if len(labelled) < 20:
        raise ValueError("At least 20 labelled users are required")

    x = np.asarray([extract_user_features(user) for user in labelled], dtype=np.float64)
    y = np.asarray([user.label for user in labelled], dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="binary", zero_division=0
    )
    metrics = {
        "dataset": "train_data.jsonl" if args.full else "train_data_sampled.jsonl",
        "records": len(labelled),
        "train_records": len(y_train),
        "test_records": len(y_test),
        "seed": args.seed,
        "feature_count": int(x.shape[1]),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
    }

    model_path = PROJECT_ROOT / "models" / "artifacts" / "account_baseline.joblib"
    metrics_path = PROJECT_ROOT / "outputs" / "baseline_metrics.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Model written to {model_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()

