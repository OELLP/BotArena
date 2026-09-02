from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from agents.adapters import augment_fusion_training, fusion_matrix, propagation_coverage, relation_coverage
from agents.neural_models import (
    GRAPH_FEATURES,
    SEQUENCE_FEATURES,
    GraphSAGEClassifier,
    TemporalGRUClassifier,
    graph_arrays,
    temporal_array,
)
from experiments.train_multiagent_baselines import evaluate
from scripts.misbot_information import InformationRecord, extract_propagation_features, iter_information
from scripts.misbot_io import DEFAULT_MISBOT_ROOT, MisBotDataset, PROJECT_ROOT


class GraphDataset(Dataset):
    def __init__(self, records: list[InformationRecord], labels: np.ndarray, indices: np.ndarray, max_nodes: int) -> None:
        self.records = records
        self.labels = labels
        self.indices = indices
        self.max_nodes = max_nodes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        nodes, edges = graph_arrays(self.records[index], self.max_nodes)
        return torch.from_numpy(nodes), torch.from_numpy(edges), int(self.labels[index])


class TemporalDataset(Dataset):
    def __init__(
        self,
        records: list[InformationRecord],
        labels: np.ndarray,
        indices: np.ndarray,
        static: np.ndarray,
        max_events: int,
    ) -> None:
        self.records = records
        self.labels = labels
        self.indices = indices
        self.static = static
        self.max_events = max_events

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        return (
            torch.from_numpy(temporal_array(self.records[index], self.max_events)),
            torch.from_numpy(self.static[index]),
            int(self.labels[index]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraphSAGE, temporal GRU and learned information fusion")
    parser.add_argument("--root", type=Path, default=DEFAULT_MISBOT_ROOT)
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "models" / "artifacts")
    parser.add_argument("--metrics-path", type=Path, default=PROJECT_ROOT / "outputs" / "information_advanced_metrics.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = MisBotDataset(args.root)
    dataset.require_complete()
    misinformation = list(iter_information(dataset, "misinformation", limit=args.limit_per_category))
    verified = list(iter_information(dataset, "verified_information", limit=args.limit_per_category))
    records = misinformation + verified
    labels = np.asarray([1] * len(misinformation) + [0] * len(verified), dtype=np.int64)
    if len(records) < 50:
        raise ValueError("At least 50 information records are required")

    indices = np.arange(len(records))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = _class_weights(labels[train_idx], device)

    graph_model = GraphSAGEClassifier(hidden_dim=args.hidden_dim).to(device)
    graph_model = _train_model(
        graph_model,
        DataLoader(
            GraphDataset(records, labels, train_idx, args.max_nodes),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=_collate_graphs,
        ),
        DataLoader(
            GraphDataset(records, labels, validation_idx, args.max_nodes),
            batch_size=args.batch_size,
            collate_fn=_collate_graphs,
        ),
        class_weights,
        args.epochs,
        args.patience,
        device,
        _graph_step,
        "GraphSAGE",
    )

    static = np.asarray([extract_propagation_features(record) for record in records], dtype=np.float32)
    static_mean = static[train_idx].mean(axis=0)
    static_scale = static[train_idx].std(axis=0)
    static_scale[static_scale < 1e-6] = 1.0
    normalized_static = (static - static_mean) / static_scale
    temporal_model = TemporalGRUClassifier(hidden_dim=args.hidden_dim).to(device)
    temporal_model = _train_model(
        temporal_model,
        DataLoader(
            TemporalDataset(records, labels, train_idx, normalized_static, args.max_events),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=_collate_temporal,
        ),
        DataLoader(
            TemporalDataset(records, labels, validation_idx, normalized_static, args.max_events),
            batch_size=args.batch_size,
            collate_fn=_collate_temporal,
        ),
        class_weights,
        args.epochs,
        args.patience,
        device,
        _temporal_step,
        "TemporalGRU",
    )

    graph_validation = _predict(
        graph_model,
        DataLoader(GraphDataset(records, labels, validation_idx, args.max_nodes), batch_size=args.batch_size, collate_fn=_collate_graphs),
        device,
        _graph_step,
    )
    graph_test = _predict(
        graph_model,
        DataLoader(GraphDataset(records, labels, test_idx, args.max_nodes), batch_size=args.batch_size, collate_fn=_collate_graphs),
        device,
        _graph_step,
    )
    temporal_validation = _predict(
        temporal_model,
        DataLoader(TemporalDataset(records, labels, validation_idx, normalized_static, args.max_events), batch_size=args.batch_size, collate_fn=_collate_temporal),
        device,
        _temporal_step,
    )
    temporal_test = _predict(
        temporal_model,
        DataLoader(TemporalDataset(records, labels, test_idx, normalized_static, args.max_events), batch_size=args.batch_size, collate_fn=_collate_temporal),
        device,
        _temporal_step,
    )

    relation_coverages = np.asarray([relation_coverage(record) for record in records])
    temporal_coverages = np.asarray([propagation_coverage(record) for record in records])
    fusion_model = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
        method="sigmoid",
        cv=2,
    )
    validation_fusion, validation_labels = augment_fusion_training(
        fusion_matrix(
            graph_validation,
            temporal_validation,
            relation_coverages[validation_idx],
            temporal_coverages[validation_idx],
        ),
        labels[validation_idx],
    )
    fusion_model.fit(validation_fusion, validation_labels)
    fused_test = fusion_model.predict_proba(
        fusion_matrix(
            graph_test,
            temporal_test,
            relation_coverages[test_idx],
            temporal_coverages[test_idx],
        )
    )[:, 1]

    artifact_root = args.artifact_dir
    artifact_root.mkdir(parents=True, exist_ok=True)
    graph_path = artifact_root / "relation_graphsage.pt"
    temporal_path = artifact_root / "propagation_gru.pt"
    torch.save(
        {
            "state_dict": graph_model.cpu().state_dict(),
            "config": {"input_dim": GRAPH_FEATURES, "hidden_dim": args.hidden_dim},
            "max_nodes": args.max_nodes,
        },
        graph_path,
    )
    torch.save(
        {
            "state_dict": temporal_model.cpu().state_dict(),
            "config": {
                "sequence_dim": SEQUENCE_FEATURES,
                "static_dim": static.shape[1],
                "hidden_dim": args.hidden_dim,
            },
            "static_mean": static_mean.tolist(),
            "static_scale": static_scale.tolist(),
            "max_events": args.max_events,
        },
        temporal_path,
    )
    joblib.dump(
        {
            "version": "advanced-v2",
            "relation_agent": {"kind": "graphsage", "model_name": "GraphSAGE", "path": graph_path.name},
            "propagation_agent": {"kind": "temporal_gru", "model_name": "GRU", "path": temporal_path.name},
            "fusion_agent": fusion_model,
        },
        artifact_root / "information_agents_advanced.joblib",
    )

    metrics = {
        "version": "advanced-v2",
        "task": "misinformation_vs_verified_information",
        "records": len(records),
        "train_records": len(train_idx),
        "validation_records": len(validation_idx),
        "test_records": len(test_idx),
        "seed": args.seed,
        "relation_agent": evaluate(labels[test_idx], graph_test),
        "propagation_agent": evaluate(labels[test_idx], temporal_test),
        "decision_agent": evaluate(labels[test_idx], fused_test),
        "models": {"relation_agent": "GraphSAGE", "propagation_agent": "GRU", "decision_agent": "learned fusion"},
    }
    metrics_path = args.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int,
    patience: int,
    device: torch.device,
    step,
    name: str,
) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = step(model, batch, device)
            loss_fn(logits, labels).backward()
            optimizer.step()
        probabilities = _predict(model, validation_loader, device, step)
        validation_labels = np.concatenate([batch[-1].numpy() for batch in validation_loader])
        auc = evaluate(validation_labels, probabilities)["roc_auc"]
        print(f"{name} epoch {epoch + 1}/{epochs}: validation AUC={auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"{name}: early stopping after {epoch + 1} epochs")
                break
    model.load_state_dict(best_state)
    return model


def _predict(model: nn.Module, loader: DataLoader, device: torch.device, step) -> np.ndarray:
    model.eval()
    probabilities: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            logits, _ = step(model, batch, device)
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def _collate_graphs(batch):
    nodes: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    graph_ids: list[torch.Tensor] = []
    labels: list[int] = []
    offset = 0
    for graph_id, (graph_nodes, graph_edges, label) in enumerate(batch):
        nodes.append(graph_nodes)
        if graph_edges.numel():
            edges.append(graph_edges + offset)
        graph_ids.append(torch.full((len(graph_nodes),), graph_id, dtype=torch.long))
        labels.append(label)
        offset += len(graph_nodes)
    edge_index = torch.cat(edges, dim=1) if edges else torch.empty((2, 0), dtype=torch.long)
    return torch.cat(nodes), edge_index, torch.cat(graph_ids), torch.tensor(labels, dtype=torch.long)


def _collate_temporal(batch):
    sequences, static, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return (
        pad_sequence(sequences, batch_first=True),
        lengths,
        torch.stack(static),
        torch.tensor(labels, dtype=torch.long),
    )


def _graph_step(model: nn.Module, batch, device: torch.device):
    nodes, edges, graph_ids, labels = batch
    return model(nodes.to(device), edges.to(device), graph_ids.to(device)), labels.to(device)


def _temporal_step(model: nn.Module, batch, device: torch.device):
    sequence, lengths, static, labels = batch
    return model(sequence.to(device), lengths.to(device), static.to(device)), labels.to(device)


def _class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2)
    return torch.as_tensor(len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)


if __name__ == "__main__":
    main()
