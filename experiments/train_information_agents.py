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

from scripts.misbot_information import extract_propagation_features, extract_relation_features, iter_information
from scripts.misbot_io import DEFAULT_MISBOT_ROOT, MisBotDataset, PROJECT_ROOT


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
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
    }


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train relation and propagation agents for information risk")
    parser.add_argument("--root", type=Path, default=DEFAULT_MISBOT_ROOT)
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    dataset = MisBotDataset(args.root)
    dataset.require_complete()
    misinformation = list(iter_information(dataset, "misinformation", limit=args.limit_per_category))
    verified = list(iter_information(dataset, "verified_information", limit=args.limit_per_category))
    records = misinformation + verified
    labels = np.asarray([1] * len(misinformation) + [0] * len(verified), dtype=np.int64)
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    relation_x = np.asarray([extract_relation_features(item) for item in records], dtype=np.float64)
    propagation_x = np.asarray([extract_propagation_features(item) for item in records], dtype=np.float64)
    relation_model = make_model(args.seed)
    propagation_model = make_model(args.seed)
    relation_model.fit(relation_x[train_idx], y_train)
    propagation_model.fit(propagation_x[train_idx], y_train)
    relation_prob = relation_model.predict_proba(relation_x[test_idx])[:, 1]
    propagation_prob = propagation_model.predict_proba(propagation_x[test_idx])[:, 1]
    fused_prob = relation_prob * 0.5 + propagation_prob * 0.5

    metrics = {
        "task": "misinformation_vs_verified_information",
        "records": len(records),
        "misinformation_records": len(misinformation),
        "verified_records": len(verified),
        "train_records": len(train_idx),
        "test_records": len(test_idx),
        "seed": args.seed,
        "weights": {"relation_agent": 0.5, "propagation_agent": 0.5},
        "relation_agent": evaluate(y_test, relation_prob),
        "propagation_agent": evaluate(y_test, propagation_prob),
        "decision_agent": evaluate(y_test, fused_prob),
    }
    artifact_path = PROJECT_ROOT / "models" / "artifacts" / "information_agents.joblib"
    metrics_path = PROJECT_ROOT / "outputs" / "information_agent_metrics.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "relation_agent": relation_model,
            "propagation_agent": propagation_model,
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

