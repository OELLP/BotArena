from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from agents.adapters import account_behavior_coverage, augment_fusion_training, fusion_matrix, text_coverage
from experiments.train_multiagent_baselines import evaluate
from scripts.misbot_io import (
    DEFAULT_MISBOT_ROOT,
    MisBotDataset,
    PROJECT_ROOT,
    extract_user_features,
    extract_user_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost, MacBERT and learned account fusion")
    parser.add_argument("--root", type=Path, default=DEFAULT_MISBOT_ROOT)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--catboost-task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "models" / "artifacts")
    parser.add_argument("--metrics-path", type=Path, default=PROJECT_ROOT / "outputs" / "multiagent_advanced_metrics.json")
    args = parser.parse_args()

    dataset = MisBotDataset(args.root)
    dataset.require_complete()
    users = [user for user in dataset.iter_users(sampled=not args.full, limit=args.limit) if user.label in (0, 1)]
    if len(users) < 50:
        raise ValueError("At least 50 labelled users are required")

    labels = np.asarray([user.label for user in users], dtype=np.int64)
    indices = np.arange(len(users))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    validation_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.5,
        random_state=args.seed,
        stratify=labels[holdout_idx],
    )

    behavior_x = np.asarray([extract_user_features(user) for user in users], dtype=np.float32)
    behavior_model = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.06,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=args.seed,
        task_type=args.catboost_task_type,
        verbose=50,
    )
    behavior_model.fit(
        behavior_x[train_idx],
        labels[train_idx],
        eval_set=(behavior_x[validation_idx], labels[validation_idx]),
        early_stopping_rounds=50,
    )

    texts = [extract_user_text(user) or "[EMPTY]" for user in users]
    artifact_root = args.artifact_dir
    text_path = artifact_root / "macbert_text"
    text_model, tokenizer, device = _train_macbert(
        [texts[index] for index in train_idx],
        labels[train_idx],
        [texts[index] for index in validation_idx],
        labels[validation_idx],
        args,
    )
    text_path.mkdir(parents=True, exist_ok=True)
    text_model.save_pretrained(text_path)
    tokenizer.save_pretrained(text_path)

    behavior_validation = behavior_model.predict_proba(behavior_x[validation_idx])[:, 1]
    behavior_test = behavior_model.predict_proba(behavior_x[test_idx])[:, 1]
    text_validation = _predict_macbert(
        text_model,
        tokenizer,
        [texts[index] for index in validation_idx],
        args.batch_size,
        args.max_length,
        device,
    )
    text_test = _predict_macbert(
        text_model,
        tokenizer,
        [texts[index] for index in test_idx],
        args.batch_size,
        args.max_length,
        device,
    )

    behavior_coverage = np.asarray([account_behavior_coverage(user) for user in users])
    semantic_coverage = np.asarray([text_coverage(user) for user in users])
    fusion_model = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
        method="sigmoid",
        cv=2,
    )
    validation_fusion, validation_labels = augment_fusion_training(
        fusion_matrix(
            behavior_validation,
            text_validation,
            behavior_coverage[validation_idx],
            semantic_coverage[validation_idx],
        ),
        labels[validation_idx],
    )
    fusion_model.fit(validation_fusion, validation_labels)
    fused_test = fusion_model.predict_proba(
        fusion_matrix(
            behavior_test,
            text_test,
            behavior_coverage[test_idx],
            semantic_coverage[test_idx],
        )
    )[:, 1]

    metrics = {
        "version": "advanced-v2",
        "dataset": "train_data.jsonl" if args.full else "train_data_sampled.jsonl",
        "records": len(users),
        "train_records": len(train_idx),
        "validation_records": len(validation_idx),
        "test_records": len(test_idx),
        "seed": args.seed,
        "behavior_agent": evaluate(labels[test_idx], behavior_test),
        "text_agent": evaluate(labels[test_idx], text_test),
        "decision_agent": evaluate(labels[test_idx], fused_test),
        "models": {"behavior_agent": "CatBoost", "text_agent": "MacBERT", "decision_agent": "learned fusion"},
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": "advanced-v2",
            "behavior_agent": {"kind": "feature", "model_name": "CatBoost", "model": behavior_model},
            "text_agent": {"kind": "macbert", "model_name": "MacBERT", "path": text_path.name},
            "fusion_agent": fusion_model,
        },
        artifact_root / "multiagent_advanced.joblib",
    )
    metrics_path = args.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _train_macbert(
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
    validation_labels: np.ndarray,
    args: argparse.Namespace,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches = DataLoader(list(zip(train_texts, train_labels.tolist())), batch_size=args.batch_size, shuffle=True)

    best_auc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for texts, labels in batches:
            encoded = tokenizer(
                list(texts),
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            targets = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**encoded, labels=targets).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        probabilities = _predict_macbert(
            model,
            tokenizer,
            validation_texts,
            args.batch_size,
            args.max_length,
            device,
        )
        auc = evaluate(validation_labels, probabilities)["roc_auc"]
        print(f"MacBERT epoch {epoch + 1}/{args.epochs}: validation AUC={auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, device


def _predict_macbert(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device) -> np.ndarray:
    import torch

    model.eval()
    output: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            output.extend(torch.softmax(model(**encoded).logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(output, dtype=np.float64)


if __name__ == "__main__":
    main()
