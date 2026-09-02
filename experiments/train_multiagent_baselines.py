from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.misbot_io import (
    DEFAULT_MISBOT_ROOT,
    MisBotDataset,
    PROJECT_ROOT,
    extract_user_features,
    extract_user_text,
)


def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predicted = (probabilities >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 6),
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text and behavior experts with decision fusion")
    parser.add_argument("--root", type=Path, default=DEFAULT_MISBOT_ROOT)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-text-features", type=int, default=30000)
    args = parser.parse_args()

    dataset = MisBotDataset(args.root)
    dataset.require_complete()
    users = [user for user in dataset.iter_users(sampled=not args.full, limit=args.limit) if user.label in (0, 1)]
    if len(users) < 20:
        raise ValueError("At least 20 labelled users are required")

    indices = np.arange(len(users))
    labels = np.asarray([user.label for user in users], dtype=np.int64)
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    behavior_x = np.asarray([extract_user_features(user) for user in users], dtype=np.float64)
    behavior_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    behavior_model.fit(behavior_x[train_idx], y_train)
    behavior_prob = behavior_model.predict_proba(behavior_x[test_idx])[:, 1]

    texts = [extract_user_text(user) for user in users]
    text_model = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=3,
                    max_features=args.max_text_features,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    text_model.fit([texts[index] for index in train_idx], y_train)
    text_prob = text_model.predict_proba([texts[index] for index in test_idx])[:, 1]

    fused_prob = behavior_prob * 0.4 + text_prob * 0.6
    metrics = {
        "dataset": "train_data.jsonl" if args.full else "train_data_sampled.jsonl",
        "records": len(users),
        "train_records": len(train_idx),
        "test_records": len(test_idx),
        "seed": args.seed,
        "weights": {"behavior_agent": 0.4, "text_agent": 0.6},
        "behavior_agent": evaluate(y_test, behavior_prob),
        "text_agent": evaluate(y_test, text_prob),
        "decision_agent": evaluate(y_test, fused_prob),
    }

    artifact_path = PROJECT_ROOT / "models" / "artifacts" / "multiagent_baselines.joblib"
    metrics_path = PROJECT_ROOT / "outputs" / "multiagent_baseline_metrics.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "behavior_agent": behavior_model,
            "text_agent": text_model,
            "weights": metrics["weights"],
        },
        artifact_path,
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Models written to {artifact_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
