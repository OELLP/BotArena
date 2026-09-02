from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import os
import re
from typing import Any

import httpx

from scripts.misbot_information import InformationRecord
from scripts.misbot_io import UserRecord


API_URL = "https://m.weibo.cn/api/container/getIndex"
DESKTOP_PROFILE_URL = "https://weibo.com/ajax/profile/info"
DESKTOP_POSTS_URL = "https://weibo.com/ajax/statuses/mymblog"
MOBILE_COMMENTS_URL = "https://m.weibo.cn/comments/hotflow"
DESKTOP_COMMENTS_URL = "https://weibo.com/ajax/statuses/buildComments"
MOBILE_REPOSTS_URL = "https://m.weibo.cn/api/statuses/repostTimeline"
DESKTOP_LIKES_URL = "https://weibo.com/ajax/statuses/likeShow"
INTERACTION_SAMPLE_LIMIT = 60
SECONDARY_INTERACTION_SAMPLE_LIMIT = 20
GRAPH_POST_LIMIT = 3
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
}
MBRANK_VALUES = tuple(range(10))
MBTYPE_VALUES = (0, 2, 11, 12, 13, 14)


class WeiboScrapeError(RuntimeError):
    pass


def normalize_uid(target: str) -> str:
    value = target.strip()
    if value.isdigit():
        return value
    match = re.search(r"(?:weibo\.com/(?:u/)?|m\.weibo\.cn/u/)(\d+)", value, re.IGNORECASE)
    if not match:
        raise ValueError("请输入数字微博UID或包含数字UID的公开主页链接")
    return match.group(1)


def fetch_weibo_account(
    target: str,
    recent_posts: int = 20,
    client: httpx.Client | None = None,
) -> tuple[
    UserRecord,
    InformationRecord | None,
    dict[str, Any],
    list[tuple[InformationRecord, dict[str, Any]]],
]:
    uid = normalize_uid(target)
    owns_client = client is None
    cookie = os.getenv("WEIBO_COOKIE", "").strip()
    headers = {**HEADERS, **({"Cookie": cookie} if cookie else {})}
    client = client or httpx.Client(headers=headers, timeout=12.0, follow_redirects=True)
    try:
        if cookie:
            return _fetch_desktop_account(client, uid, recent_posts)
        profile_payload = _get_json(
            client,
            {"type": "uid", "value": uid, "containerid": f"100505{uid}"},
            uid,
        )
        profile = profile_payload.get("data", {}).get("userInfo") or {}
        if not profile:
            raise WeiboScrapeError("未读取到该账号的公开资料")

        posts: list[str] = []
        interaction_samples: list[tuple[InformationRecord, dict[str, Any]]] = []
        for page in range(1, 11):
            timeline = _get_json(
                client,
                {
                    "type": "uid",
                    "value": uid,
                    "containerid": f"107603{uid}",
                    "page": page,
                },
                uid,
            )
            cards = timeline.get("data", {}).get("cards") or []
            for card in cards:
                if card.get("card_type") != 9 or not isinstance(card.get("mblog"), dict):
                    continue
                mblog = card["mblog"]
                post = _clean_text(mblog.get("text_raw") or mblog.get("text") or "")
                repost = mblog.get("retweeted_status") or {}
                repost_text = _clean_text(repost.get("text_raw") or repost.get("text") or "")
                combined = " ".join(part for part in (post, repost_text) if part).strip()
                if combined:
                    posts.append(combined)
                    if len(interaction_samples) < GRAPH_POST_LIMIT:
                        interaction_samples.append(_post_information(mblog, combined, uid))
                if len(posts) >= recent_posts:
                    break
            if len(posts) >= recent_posts or not cards:
                break

        record = UserRecord(
            uid=str(profile.get("id") or uid),
            tweets=posts[:recent_posts],
            description=_clean_text(profile.get("description") or ""),
            numerical=[
                _count(profile.get("followers_count")),
                _count(profile.get("follow_count")),
                _count(profile.get("statuses_count")),
            ],
            categorical=_misbot_categories(profile),
            label=None,
        )
        for index, (information, post_summary) in enumerate(interaction_samples):
            limit = INTERACTION_SAMPLE_LIMIT if index == 0 else SECONDARY_INTERACTION_SAMPLE_LIMIT
            _enrich_interactions(client, information, post_summary, uid, desktop=False, limit=limit)
        latest_information, latest_post = interaction_samples[0] if interaction_samples else (None, None)
        return record, latest_information, _profile_summary(record, profile, latest_post), interaction_samples
    finally:
        if owns_client:
            client.close()


def _get_json(client: httpx.Client, params: dict[str, Any], uid: str) -> dict[str, Any]:
    try:
        response = client.get(API_URL, params=params, headers={"Referer": f"https://m.weibo.cn/u/{uid}"})
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 432:
            raise WeiboScrapeError(
                "微博拒绝匿名抓取（HTTP 432）。请在启动后端前设置你本人账号的 WEIBO_COOKIE 环境变量。"
            ) from exc
        raise WeiboScrapeError(f"微博公开页面返回HTTP {exc.response.status_code}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise WeiboScrapeError("微博公开页面访问失败，请稍后重试") from exc
    if payload.get("ok") != 1:
        message = str(payload.get("msg") or "页面可能需要登录或触发访问限制")
        raise WeiboScrapeError(f"微博未返回公开数据：{message}")
    return payload


def _fetch_desktop_account(
    client: httpx.Client,
    uid: str,
    recent_posts: int,
) -> tuple[
    UserRecord,
    InformationRecord | None,
    dict[str, Any],
    list[tuple[InformationRecord, dict[str, Any]]],
]:
    profile_payload = _desktop_json(client, DESKTOP_PROFILE_URL, {"uid": uid}, uid)
    profile = profile_payload.get("data", {}).get("user") or {}
    if not profile:
        raise WeiboScrapeError("登录会话未读取到该账号资料，请更新 WEIBO_COOKIE")

    posts: list[str] = []
    interaction_samples: list[tuple[InformationRecord, dict[str, Any]]] = []
    for page in range(1, 11):
        timeline = _desktop_json(
            client,
            DESKTOP_POSTS_URL,
            {"uid": uid, "page": page, "feature": 0},
            uid,
        )
        items = timeline.get("data", {}).get("list") or []
        for item in items:
            post = _clean_text(item.get("text_raw") or item.get("text") or "")
            repost = item.get("retweeted_status") or {}
            repost_text = _clean_text(repost.get("text_raw") or repost.get("text") or "")
            combined = " ".join(part for part in (post, repost_text) if part).strip()
            if combined:
                posts.append(combined)
                if len(interaction_samples) < GRAPH_POST_LIMIT:
                    interaction_samples.append(_post_information(item, combined, uid))
            if len(posts) >= recent_posts:
                break
        if len(posts) >= recent_posts or not items:
            break

    record = UserRecord(
        uid=str(profile.get("idstr") or profile.get("id") or uid),
        tweets=posts[:recent_posts],
        description=_clean_text(profile.get("description") or ""),
        numerical=[
            _count(profile.get("followers_count")),
            _count(profile.get("friends_count") or profile.get("follow_count")),
            _count(profile.get("statuses_count")),
        ],
        categorical=_misbot_categories(profile),
        label=None,
    )
    for index, (information, post_summary) in enumerate(interaction_samples):
        limit = INTERACTION_SAMPLE_LIMIT if index == 0 else SECONDARY_INTERACTION_SAMPLE_LIMIT
        _enrich_interactions(client, information, post_summary, uid, desktop=True, limit=limit)
    latest_information, latest_post = interaction_samples[0] if interaction_samples else (None, None)
    return record, latest_information, _profile_summary(record, profile, latest_post), interaction_samples


def _desktop_json(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    uid: str,
) -> dict[str, Any]:
    try:
        response = client.get(url, params=params, headers={"Referer": f"https://weibo.com/u/{uid}"})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise WeiboScrapeError("微博登录会话访问失败，请检查 WEIBO_COOKIE") from exc
    if payload.get("ok") != 1:
        raise WeiboScrapeError(str(payload.get("message") or "微博登录会话已失效或访问受限"))
    return payload


def _profile_summary(
    record: UserRecord,
    profile: dict[str, Any],
    latest_post: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "uid": record.uid,
        "screen_name": str(profile.get("screen_name") or ""),
        "description": record.description,
        "followers_count": int(record.numerical[0]),
        "follow_count": int(record.numerical[1]),
        "statuses_count": int(record.numerical[2]),
        "verified": _flag(profile.get("verified")),
        "svip": _flag(profile.get("svip")),
        "mbrank": int(_count(profile.get("mbrank"))),
        "mbtype": int(_count(profile.get("mb_type_name") or profile.get("mbtype"))),
        "posts_collected": len(record.tweets),
        "latest_post": latest_post,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _post_information(
    post: dict[str, Any],
    content: str,
    uid: str,
) -> tuple[InformationRecord, dict[str, Any]]:
    target_id = str(post.get("idstr") or post.get("id") or post.get("mid") or f"{uid}_latest")
    record = InformationRecord(
        category="unknown",
        content=content,
        publish_time=None,
        repost_count=int(_count(post.get("reposts_count"))),
        comment_count=int(_count(post.get("comments_count"))),
        attitude_count=int(_count(post.get("attitudes_count"))),
        comment_users=[],
        repost_users=[],
        attitude_users=[],
        comment_nodes=0,
        comment_edges=0,
        repost_nodes=0,
        repost_edges=0,
        activity_times=[],
    )
    return record, {
        "id": target_id,
        "content": content,
        "repost_count": record.repost_count,
        "comment_count": record.comment_count,
        "attitude_count": record.attitude_count,
    }


def _enrich_interactions(
    client: httpx.Client,
    record: InformationRecord | None,
    latest_post: dict[str, Any] | None,
    uid: str,
    *,
    desktop: bool,
    limit: int,
) -> None:
    """Best-effort sampling; blocked interaction endpoints must not break account detection."""
    if record is None or not latest_post:
        return
    _enrich_comments(client, record, latest_post, uid, desktop=desktop, limit=limit)
    post_id = str(latest_post["id"])
    referer = f"https://weibo.com/{uid}/{post_id}"
    _enrich_paged_users(
        client,
        record,
        MOBILE_REPOSTS_URL,
        {"id": post_id, "count": 20},
        referer,
        "repost",
        limit,
    )
    _enrich_paged_users(
        client,
        record,
        DESKTOP_LIKES_URL,
        {"id": post_id, "attitude_type": 0, "attitude_enable": 1},
        referer,
        "attitude",
        limit,
    )
    record.interaction_edges = list(dict.fromkeys(record.interaction_edges))
    record.comment_nodes = len(record.comment_users) + (1 if record.comment_users else 0)
    record.comment_edges = sum(edge[2] == "comment" for edge in record.interaction_edges)
    record.repost_nodes = len(record.repost_users) + (1 if record.repost_users else 0)
    record.repost_edges = sum(edge[2] == "repost" for edge in record.interaction_edges)


def _enrich_comments(
    client: httpx.Client,
    record: InformationRecord,
    latest_post: dict[str, Any],
    uid: str,
    *,
    desktop: bool,
    limit: int,
) -> None:
    post_id = str(latest_post["id"])
    if desktop:
        url = DESKTOP_COMMENTS_URL
        base_params = {
            "id": post_id,
            "flow": 0,
            "is_reload": 1,
            "is_show_bulletin": 2,
            "count": 20,
            "uid": uid,
        }
        referer = f"https://weibo.com/{uid}/{post_id}"
    else:
        url = MOBILE_COMMENTS_URL
        base_params = {"id": post_id, "mid": post_id, "max_id_type": 0}
        referer = f"https://m.weibo.cn/detail/{post_id}"
    max_id: Any = None
    for page in range((limit + 19) // 20):
        params = dict(base_params)
        if desktop:
            params.update({"is_mix": int(page > 0), "max_id": max_id or 0})
        elif max_id is not None:
            params["max_id"] = max_id
        payload = _optional_json(client, url, params, referer)
        items, metadata = _payload_items(payload)
        if not items:
            break
        _append_interactions(record, items, "comment", limit)
        if len(record.comment_users) >= limit:
            break
        next_id = metadata.get("max_id")
        if not next_id or next_id == max_id:
            break
        max_id = next_id
        if not desktop and "max_id_type" in metadata:
            base_params["max_id_type"] = metadata["max_id_type"]


def _enrich_paged_users(
    client: httpx.Client,
    record: InformationRecord,
    url: str,
    base_params: dict[str, Any],
    referer: str,
    action: str,
    limit: int,
) -> None:
    users = getattr(record, f"{action}_users")
    for page in range(1, (limit + 19) // 20 + 1):
        payload = _optional_json(client, url, {**base_params, "page": page}, referer)
        items, _ = _payload_items(payload)
        if not items:
            break
        _append_interactions(record, items, action, limit)
        if len(users) >= limit or len(items) < 20:
            break


def _payload_items(payload: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = (payload or {}).get("data") or {}
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)], payload or {}
    if not isinstance(data, dict):
        return [], {}
    for key in ("data", "list", "statuses"):
        if isinstance(data.get(key), list):
            return [item for item in data[key] if isinstance(item, dict)], data
    return [], data


def _append_interactions(
    record: InformationRecord,
    items: list[dict[str, Any]],
    action: str,
    limit: int,
) -> None:
    users = getattr(record, f"{action}_users")
    for item in items:
        source = item.get("mblog") if isinstance(item.get("mblog"), dict) else item
        user = source.get("user") or item.get("user") or {}
        uid = str(user.get("idstr") or user.get("id") or "")
        if not uid:
            continue
        if uid not in users:
            users.append(uid)
        record.interaction_edges.append(("__source__", uid, action))
        if (timestamp := _timestamp(source.get("created_at"))) is not None:
            record.activity_times.append(timestamp)
            record.activity_events.append((timestamp, action))
        if len(users) >= limit:
            break


def _optional_json(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    referer: str,
) -> dict[str, Any] | None:
    try:
        response = client.get(url, params=params, headers={"Referer": referer})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return payload if payload.get("ok") == 1 else None


def _timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return parsedate_to_datetime(str(value)).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _clean_text(value: Any) -> str:
    text = re.sub(r"<br\s*/?>", "\n", str(value), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", text))).strip()


def _misbot_categories(profile: dict[str, Any]) -> list[float]:
    verified = _flag(profile.get("verified"))
    svip = _flag(profile.get("svip"))
    mbrank = int(_count(profile.get("mbrank")))
    mbtype = int(_count(profile.get("mb_type_name") or profile.get("mbtype")))
    return (
        [1.0, 0.0] if verified else [0.0, 1.0]
    ) + (
        [1.0, 0.0] if svip else [0.0, 1.0]
    ) + [float(mbrank == value) for value in MBRANK_VALUES] + [
        float(mbtype == value) for value in MBTYPE_VALUES
    ]


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _count(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(max(value, 0))
    text = str(value or "0").replace(",", "").strip()
    multiplier = 100_000_000 if text.endswith("亿") else 10_000 if text.endswith("万") else 1
    try:
        return max(0.0, float(text.rstrip("万亿")) * multiplier)
    except ValueError:
        return 0.0
