from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents import BotArenaRuntime
from scripts.misbot_information import InformationRecord, iter_information
from scripts.misbot_io import MisBotDataset, PROJECT_ROOT, UserRecord
from scripts.weibo_scraper import (
    INTERACTION_SAMPLE_LIMIT,
    SECONDARY_INTERACTION_SAMPLE_LIMIT,
    WeiboScrapeError,
    fetch_weibo_account,
)


class AccountRequest(BaseModel):
    uid: str = Field(min_length=1)
    tweets: list[str] = Field(default_factory=list)
    description: str = ""
    numerical: list[float] = Field(default_factory=list)
    categorical: list[float] = Field(default_factory=list)


class InformationRequest(BaseModel):
    target_id: str = Field(min_length=1)
    category: str = "unknown"
    content: str = ""
    publish_time: float | None = None
    repost_count: int = Field(default=0, ge=0)
    comment_count: int = Field(default=0, ge=0)
    attitude_count: int = Field(default=0, ge=0)
    comment_users: list[str] = Field(default_factory=list)
    repost_users: list[str] = Field(default_factory=list)
    attitude_users: list[str] = Field(default_factory=list)
    comment_nodes: int = Field(default=0, ge=0)
    comment_edges: int = Field(default=0, ge=0)
    repost_nodes: int = Field(default=0, ge=0)
    repost_edges: int = Field(default=0, ge=0)
    activity_times: list[float] = Field(default_factory=list)
    interaction_edges: list[tuple[str, str, str]] = Field(default_factory=list)
    activity_events: list[tuple[float, str]] = Field(default_factory=list)


class WeiboAccountRequest(BaseModel):
    target: str = Field(min_length=1, max_length=200)
    recent_posts: int = Field(default=20, ge=1, le=100)


app = FastAPI(title="BotArena API", version="0.2.0", description="微博社交机器人异构多智能体检测接口")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def runtime() -> BotArenaRuntime:
    return BotArenaRuntime()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "BotArena", "version": app.version}


@app.get("/api/metrics")
def metrics() -> dict[str, object]:
    names = (
        "misbot_profile",
        "multiagent_baseline_metrics",
        "information_agent_metrics",
        "multiagent_advanced_metrics",
        "information_advanced_metrics",
    )
    return {name: _read_json(PROJECT_ROOT / "outputs" / f"{name}.json") for name in names}


@app.post("/api/detect/account")
def detect_account(payload: AccountRequest) -> dict[str, object]:
    return runtime().analyze_user(UserRecord(**payload.model_dump(), label=None))


@app.post("/api/detect/weibo")
@app.post("/api/detect/account/weibo", include_in_schema=False)
def detect_weibo_account(payload: WeiboAccountRequest) -> dict[str, object]:
    try:
        user, information, profile, interaction_samples = fetch_weibo_account(
            payload.target, payload.recent_posts
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WeiboScrapeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    detector = runtime()
    latest_post = profile.get("latest_post")
    account_result = detector.analyze_user(user)
    information_result = (
        detector.analyze_information(information, str(latest_post["id"]))
        if information is not None and latest_post
        else None
    )
    return {
        "source": "weibo_public_web",
        "profile": profile,
        "account": account_result,
        "information": information_result,
        "interaction_graph": _interaction_graph(user, profile, interaction_samples, information_result),
        "notice": "账号结果基于公开资料与近期发文；信息结果综合最近微博的互动计数及可访问评论样本，数据覆盖率较低时应优先人工复核。",
    }


@app.post("/api/detect/information")
def detect_information(payload: InformationRequest) -> dict[str, object]:
    values = payload.model_dump()
    target_id = values.pop("target_id")
    return runtime().analyze_information(InformationRecord(**values), target_id)


@app.get("/api/demo")
def demo() -> dict[str, object]:
    dataset = MisBotDataset()
    user = next(dataset.iter_users(sampled=True, limit=1))
    information = next(iter_information(dataset, "misinformation", limit=1))
    detector = runtime()
    return {
        "account": detector.analyze_user(user),
        "information": detector.analyze_information(information, "misinformation_0000"),
    }


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _interaction_graph(
    user: UserRecord,
    profile: dict[str, object],
    samples: list[tuple[InformationRecord, dict[str, object]]],
    information_result: dict[str, object] | None,
) -> dict[str, object] | None:
    """Build a privacy-aware account-post-user propagation graph."""
    if not samples:
        return None

    account_id = f"account:{user.uid}"
    nodes: list[dict[str, object]] = [{
        "id": account_id,
        "name": str(profile.get("screen_name") or user.uid),
        "type": "account",
        "degree": 0,
        "description": user.description,
    }]
    links: list[dict[str, str]] = []
    seen_links: set[tuple[str, str, str]] = set()
    actions_by_user: dict[str, set[str]] = {}
    for index, (item, post) in enumerate(samples, start=1):
        post_id = f"post:{post['id']}"
        content = str(post.get("content") or item.content)
        nodes.append({
            "id": post_id,
            "name": f"微博 {index}",
            "type": "post",
            "degree": 0,
            "content": content,
            "repost_count": item.repost_count,
            "comment_count": item.comment_count,
            "attitude_count": item.attitude_count,
        })
        links.append({"source": account_id, "target": post_id, "action": "publish"})
        for action, users in (
            ("comment", item.comment_users),
            ("repost", item.repost_users),
            ("attitude", item.attitude_users),
        ):
            for uid in users:
                uid = str(uid)
                if not uid or uid == "__source__":
                    continue
                user_id = f"user:{uid}"
                actions_by_user.setdefault(uid, set()).add(action)
                edge = (post_id, user_id, action)
                if edge not in seen_links:
                    seen_links.add(edge)
                    links.append({"source": edge[0], "target": edge[1], "action": action})

    degree: dict[str, int] = {}
    for link in links:
        degree[link["source"]] = degree.get(link["source"], 0) + 1
        degree[link["target"]] = degree.get(link["target"], 0) + 1
    for node in nodes:
        node["degree"] = degree.get(str(node["id"]), 0)
    for uid, actions in actions_by_user.items():
        nodes.append(
            {
                "id": f"user:{uid}",
                "name": _masked_user(uid),
                "type": "multi" if len(actions) > 1 else next(iter(actions)),
                "actions": sorted(actions),
                "degree": degree.get(f"user:{uid}", 0),
            }
        )

    coverage = 0.0
    if information_result:
        values = information_result.get("data_coverage")
        if isinstance(values, dict):
            coverage = float(values.get("relation_agent", 0.0))

    return {
        "nodes": nodes,
        "links": links,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(links),
            "post_count": len(samples),
            "user_count": len(actions_by_user),
            "comment_users": sum("comment" in actions for actions in actions_by_user.values()),
            "repost_users": sum("repost" in actions for actions in actions_by_user.values()),
            "attitude_users": sum("attitude" in actions for actions in actions_by_user.values()),
            "multi_users": sum(len(actions) > 1 for actions in actions_by_user.values()),
            "coverage": round(coverage, 6),
            "sample_limit": INTERACTION_SAMPLE_LIMIT,
            "secondary_sample_limit": SECONDARY_INTERACTION_SAMPLE_LIMIT,
        },
    }


def _masked_user(uid: str) -> str:
    return f"用户…{uid[-4:]}" if len(uid) > 4 else f"用户{uid}"
