from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from agents.adapters import (
    AgentPrediction,
    account_behavior_coverage,
    behavior_evidence,
    build_adapter,
    fuse_predictions,
    propagation_coverage,
    propagation_evidence,
    relation_coverage,
    relation_evidence,
    text_coverage,
)
from scripts.misbot_information import InformationRecord, extract_propagation_features, extract_relation_features
from scripts.misbot_io import PROJECT_ROOT, UserRecord, extract_user_features, extract_user_text


class BotArenaRuntime:
    """Load interchangeable model adapters behind one account/information interface."""

    def __init__(self, artifact_dir: str | Path = PROJECT_ROOT / "models" / "artifacts") -> None:
        root = Path(artifact_dir)
        account_path = _preferred(root, "multiagent_advanced.joblib", "multiagent_baselines.joblib")
        information_path = _preferred(root, "information_agents_advanced.joblib", "information_agents.joblib")
        self.account_bundle = joblib.load(account_path)
        self.information_bundle = joblib.load(information_path)
        self.account_version = str(self.account_bundle.get("version", "baseline-v1"))
        self.information_version = str(self.information_bundle.get("version", "baseline-v1"))

        self.account_agents = {
            "behavior_agent": build_adapter(
                self.account_bundle["behavior_agent"],
                artifact_root=root,
                feature_fn=extract_user_features,
                coverage_fn=account_behavior_coverage,
                legacy_name="LogisticRegression",
                evidence_fn=behavior_evidence,
            ),
            "text_agent": build_adapter(
                self.account_bundle["text_agent"],
                artifact_root=root,
                feature_fn=extract_user_text,
                coverage_fn=text_coverage,
                legacy_name="TF-IDF + LogisticRegression",
                evidence_fn=lambda user, _features, _model: [f"分析简介与{len(user.tweets)}条近期发文"],
            ),
        }
        self.information_agents = {
            "relation_agent": build_adapter(
                self.information_bundle["relation_agent"],
                artifact_root=root,
                feature_fn=extract_relation_features,
                coverage_fn=relation_coverage,
                legacy_name="LogisticRegression",
                evidence_fn=relation_evidence,
            ),
            "propagation_agent": build_adapter(
                self.information_bundle["propagation_agent"],
                artifact_root=root,
                feature_fn=extract_propagation_features,
                coverage_fn=propagation_coverage,
                legacy_name="LogisticRegression",
                evidence_fn=propagation_evidence,
            ),
        }

    def analyze_user(self, user: UserRecord) -> dict[str, Any]:
        predictions = {name: agent.predict(user) for name, agent in self.account_agents.items()}
        score = _fuse_bundle(predictions, self.account_bundle)
        return _result("account", user.uid, predictions, score, self.account_version)

    def analyze_information(self, item: InformationRecord, target_id: str) -> dict[str, Any]:
        predictions = {name: agent.predict(item) for name, agent in self.information_agents.items()}
        score = _fuse_bundle(predictions, self.information_bundle)
        return _result("information", target_id, predictions, score, self.information_version)


def _preferred(root: Path, advanced_name: str, baseline_name: str) -> Path:
    advanced = root / advanced_name
    return advanced if advanced.is_file() else root / baseline_name


def _fuse_bundle(predictions: dict[str, AgentPrediction], bundle: dict[str, Any]) -> float:
    fusion_model = bundle.get("fusion_agent")
    if fusion_model is not None:
        return fuse_predictions(predictions, fusion_model)
    weights = bundle.get("weights")
    if weights:
        return sum(predictions[name].score * float(weight) for name, weight in weights.items())
    return fuse_predictions(predictions)


def _result(
    target_type: str,
    target_id: str,
    agents: dict[str, AgentPrediction],
    score: float,
    model_version: str,
) -> dict[str, Any]:
    ranked = sorted(agents.items(), key=lambda item: item[1].score, reverse=True)
    evidence = [line for _, prediction in ranked for line in prediction.evidence]
    return {
        "target_type": target_type,
        "target_id": target_id,
        "risk_score": round(score, 6),
        "risk_level": risk_level(score),
        "model_version": model_version,
        "agent_scores": {name: round(value.score, 6) for name, value in agents.items()},
        "agent_confidence": {name: round(value.confidence, 6) for name, value in agents.items()},
        "data_coverage": {name: round(value.coverage, 6) for name, value in agents.items()},
        "agent_models": {name: value.model_name for name, value in agents.items()},
        "evidence": evidence or ["当前输入未形成可复核证据"],
    }


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"
