from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from scripts.misbot_information import InformationRecord, extract_propagation_features
from scripts.misbot_io import USER_FEATURE_NAMES, UserRecord, extract_user_text


@dataclass(slots=True)
class AgentPrediction:
    score: float
    confidence: float
    coverage: float
    model_name: str
    evidence: list[str]


class AgentAdapter(Protocol):
    def predict(self, record: Any) -> AgentPrediction: ...


class FeatureAgentAdapter:
    def __init__(
        self,
        model: Any,
        feature_fn: Callable[[Any], Any],
        coverage_fn: Callable[[Any], float],
        model_name: str,
        evidence_fn: Callable[[Any, list[float], Any], list[str]],
    ) -> None:
        self.model = model
        self.feature_fn = feature_fn
        self.coverage_fn = coverage_fn
        self.model_name = model_name
        self.evidence_fn = evidence_fn

    def predict(self, record: Any) -> AgentPrediction:
        features = self.feature_fn(record)
        score = float(self.model.predict_proba([features])[0, 1])
        return AgentPrediction(
            score=score,
            confidence=_probability_confidence(score),
            coverage=self.coverage_fn(record),
            model_name=self.model_name,
            evidence=self.evidence_fn(record, features, self.model),
        )


class MacBertAgentAdapter:
    def __init__(self, model_path: Path, model_name: str = "MacBERT") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.model_name = model_name

    def predict(self, record: UserRecord) -> AgentPrediction:
        text = extract_user_text(record)
        coverage = text_coverage(record)
        if not text:
            return AgentPrediction(0.5, 0.0, 0.0, self.model_name, ["未采集到可分析文本"])
        encoded = self.tokenizer(
            text,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            score = float(self.torch.softmax(self.model(**encoded).logits, dim=1)[0, 1].item())
        return AgentPrediction(
            score,
            _probability_confidence(score),
            coverage,
            self.model_name,
            [f"对简介与{len(record.tweets)}条近期发文进行了上下文语义分析"],
        )


class GraphSAGEAgentAdapter:
    def __init__(self, checkpoint_path: Path, model_name: str = "GraphSAGE") -> None:
        import torch

        from agents.neural_models import GraphSAGEClassifier

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model = GraphSAGEClassifier(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.max_nodes = int(checkpoint.get("max_nodes", 256))
        self.model_name = model_name

    def predict(self, record: InformationRecord) -> AgentPrediction:
        from agents.neural_models import graph_arrays

        nodes, edge_index = graph_arrays(record, self.max_nodes)
        torch = self.torch
        with torch.inference_mode():
            logits = self.model(
                torch.as_tensor(nodes, device=self.device),
                torch.as_tensor(edge_index, device=self.device),
                torch.zeros(len(nodes), dtype=torch.long, device=self.device),
            )
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
        return AgentPrediction(
            score,
            _probability_confidence(score),
            relation_coverage(record),
            self.model_name,
            [
                f"分析{len(record.participants)}名参与用户和{len(record.interaction_edges)}条显式互动边",
                f"评论图与转发图共{record.comment_nodes + record.repost_nodes}个节点",
            ],
        )


class TemporalGRUAgentAdapter:
    def __init__(self, checkpoint_path: Path, model_name: str = "GRU") -> None:
        import torch

        from agents.neural_models import TemporalGRUClassifier

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model = TemporalGRUClassifier(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.static_mean = np.asarray(checkpoint["static_mean"], dtype=np.float32)
        self.static_scale = np.asarray(checkpoint["static_scale"], dtype=np.float32)
        self.max_events = int(checkpoint.get("max_events", 256))
        self.model_name = model_name

    def predict(self, record: InformationRecord) -> AgentPrediction:
        from agents.neural_models import temporal_array

        torch = self.torch
        sequence = temporal_array(record, self.max_events)
        static = (np.asarray(extract_propagation_features(record), dtype=np.float32) - self.static_mean) / self.static_scale
        with torch.inference_mode():
            logits = self.model(
                torch.as_tensor(sequence[None, ...], device=self.device),
                torch.as_tensor([len(sequence)], dtype=torch.long, device=self.device),
                torch.as_tensor(static[None, ...], device=self.device),
            )
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
        return AgentPrediction(
            score,
            _probability_confidence(score),
            propagation_coverage(record),
            self.model_name,
            [
                f"分析{len(record.activity_events or record.activity_times)}个带时间信息的传播事件",
                f"当前转评赞总量为{record.repost_count + record.comment_count + record.attitude_count}",
            ],
        )


def build_adapter(
    value: Any,
    *,
    artifact_root: Path,
    feature_fn: Callable[[Any], Any],
    coverage_fn: Callable[[Any], float],
    legacy_name: str,
    evidence_fn: Callable[[Any, list[float], Any], list[str]],
) -> AgentAdapter:
    if not isinstance(value, dict) or "kind" not in value:
        return FeatureAgentAdapter(value, feature_fn, coverage_fn, legacy_name, evidence_fn)
    kind = value["kind"]
    model_name = str(value.get("model_name") or kind)
    if kind == "feature":
        return FeatureAgentAdapter(value["model"], feature_fn, coverage_fn, model_name, evidence_fn)
    path = artifact_root / value["path"]
    if kind == "macbert":
        return MacBertAgentAdapter(path, model_name)
    if kind == "graphsage":
        return GraphSAGEAgentAdapter(path, model_name)
    if kind == "temporal_gru":
        return TemporalGRUAgentAdapter(path, model_name)
    raise ValueError(f"Unsupported agent kind: {kind}")


def fuse_predictions(
    predictions: dict[str, AgentPrediction],
    fusion_model: Any | None = None,
) -> float:
    ordered = list(predictions.values())
    if fusion_model is not None:
        features = [
            *[item.score for item in ordered],
            *[item.confidence for item in ordered],
            *[item.coverage for item in ordered],
        ]
        return float(fusion_model.predict_proba([features])[0, 1])
    weights = [max(item.coverage, 0.05) * (0.25 + 0.75 * item.confidence) for item in ordered]
    return float(np.average([item.score for item in ordered], weights=weights))


def fusion_matrix(
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    first_coverage: np.ndarray,
    second_coverage: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            first_scores,
            second_scores,
            np.abs(first_scores - 0.5) * 2,
            np.abs(second_scores - 0.5) * 2,
            first_coverage,
            second_coverage,
        )
    )


def augment_fusion_training(matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_missing = matrix.copy()
    first_missing[:, [0, 2, 4]] = (0.5, 0.0, 0.0)
    second_missing = matrix.copy()
    second_missing[:, [1, 3, 5]] = (0.5, 0.0, 0.0)
    return np.vstack((matrix, first_missing, second_missing)), np.tile(labels, 3)


def account_behavior_coverage(record: UserRecord) -> float:
    return min(
        1.0,
        0.4 * min(len(record.numerical) / 3, 1.0)
        + 0.4 * min(len(record.categorical) / 20, 1.0)
        + 0.2 * min(len(record.tweets) / 5, 1.0),
    )


def text_coverage(record: UserRecord) -> float:
    text = extract_user_text(record)
    if not text:
        return 0.0
    return min(1.0, 0.6 * len(text) / 300 + 0.4 * len(record.tweets) / 10)


def relation_coverage(record: InformationRecord) -> float:
    if record.interaction_edges:
        return min(1.0, 0.6 + len(record.interaction_edges) / 100)
    if record.participants:
        return min(0.6, 0.25 + len(record.participants) / 100)
    return 0.2 if record.comment_nodes + record.repost_nodes else 0.05


def propagation_coverage(record: InformationRecord) -> float:
    event_count = len(record.activity_events or record.activity_times)
    if event_count:
        return min(1.0, 0.4 + event_count / 100)
    total = record.repost_count + record.comment_count + record.attitude_count
    return 0.25 if total else 0.05


def behavior_evidence(record: UserRecord, features: list[float], model: Any) -> list[str]:
    try:
        from catboost import Pool

        contributions = model.get_feature_importance(Pool([features]), type="ShapValues")[0][:-1]
        positive = sorted(
            ((float(value), USER_FEATURE_NAMES[index]) for index, value in enumerate(contributions) if value > 0),
            reverse=True,
        )[:3]
        if positive:
            return ["推动风险上升的行为特征：" + "、".join(name for _, name in positive)]
    except (ImportError, AttributeError, IndexError, ValueError):
        pass
    return [f"分析{len(features)}维账号属性与行为特征"]


def relation_evidence(record: InformationRecord, _features: list[float], _model: Any) -> list[str]:
    return [f"分析{len(record.participants)}名参与用户及互动关系结构"]


def propagation_evidence(record: InformationRecord, _features: list[float], _model: Any) -> list[str]:
    return [f"分析{len(record.activity_times)}个时间点及转评赞传播规模"]


def _probability_confidence(score: float) -> float:
    return min(1.0, max(0.0, abs(score - 0.5) * 2))
