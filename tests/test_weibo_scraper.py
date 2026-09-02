from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from scripts.weibo_scraper import fetch_weibo_account, normalize_uid


class WeiboScraperTests(unittest.TestCase):
    def test_public_profile_and_posts_become_user_record(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/comments/hotflow"):
                uid = "u-comment-2" if request.url.params.get("max_id") else "u-comment-1"
                return httpx.Response(
                    200,
                    json={
                        "ok": 1,
                        "data": {
                            "data": [
                                {
                                    "user": {"id": uid},
                                    "created_at": "Wed Aug 19 12:00:00 +0800 2026",
                                }
                            ],
                            "max_id": 0 if uid.endswith("2") else 2,
                            "max_id_type": 0,
                        },
                    },
                )
            if request.url.path.endswith("/api/statuses/repostTimeline"):
                return httpx.Response(200, json={"ok": 1, "data": {"data": [{"user": {"id": "u-repost"}}]}})
            if request.url.path.endswith("/ajax/statuses/likeShow"):
                return httpx.Response(200, json={"ok": 1, "data": {"list": [{"user": {"id": "u-like"}}]}})
            container = request.url.params["containerid"]
            if container.startswith("100505"):
                return httpx.Response(
                    200,
                    json={
                        "ok": 1,
                        "data": {
                            "userInfo": {
                                "id": "123456",
                                "screen_name": "测试账号",
                                "description": "公开简介",
                                "followers_count": 100,
                                "follow_count": 20,
                                "statuses_count": 30,
                                "verified": True,
                                "svip": 1,
                                "mbrank": 7,
                                "mbtype": 12,
                            }
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "ok": 1,
                    "data": {
                        "cards": [
                            {
                                "card_type": 9,
                                "mblog": {
                                    "id": "post-1",
                                    "text": "第一条<br />微博",
                                    "reposts_count": 8,
                                    "comments_count": 3,
                                    "attitudes_count": 20,
                                },
                            },
                            {"card_type": 9, "mblog": {"id": "post-2", "text": "<a>第二条</a>"}},
                        ]
                    },
                },
            )

        with patch.dict("os.environ", {}, clear=True):
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                user, information, profile, samples = fetch_weibo_account("https://weibo.com/u/123456", 2, client)

        self.assertEqual(user.uid, "123456")
        self.assertEqual(user.tweets, ["第一条 微博", "第二条"])
        self.assertEqual(user.numerical, [100.0, 20.0, 30.0])
        self.assertEqual(user.categorical[:4], [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(user.categorical[4:14], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(user.categorical[14:], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(profile["posts_collected"], 2)
        self.assertEqual(profile["mbrank"], 7)
        self.assertEqual(profile["mbtype"], 12)
        self.assertEqual(profile["latest_post"]["id"], "post-1")
        self.assertEqual(information.repost_count, 8)
        self.assertEqual(information.comment_users, ["u-comment-1", "u-comment-2"])
        self.assertEqual(information.repost_users, ["u-repost"])
        self.assertEqual(information.attitude_users, ["u-like"])
        self.assertEqual(len(information.interaction_edges), 4)
        self.assertEqual(len(samples), 2)

    def test_rejects_non_numeric_profile_target(self) -> None:
        with self.assertRaises(ValueError):
            normalize_uid("https://weibo.com/not-a-numeric-id")

    def test_logged_in_desktop_pages_are_supported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/ajax/profile/info"):
                return httpx.Response(
                    200,
                    json={
                        "ok": 1,
                        "data": {
                            "user": {
                                "idstr": "123456",
                                "screen_name": "测试账号",
                                "description": "简介",
                                "followers_count": 12,
                                "friends_count": 3,
                                "statuses_count": 9,
                            }
                        },
                    },
                )
            return httpx.Response(
                200,
                json={"ok": 1, "data": {"list": [{"text_raw": "公开博文"}]}},
            )

        with patch.dict("os.environ", {"WEIBO_COOKIE": "SUB=test"}):
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                user, information, profile, samples = fetch_weibo_account("123456", 1, client)

        self.assertEqual(user.tweets, ["公开博文"])
        self.assertEqual(profile["posts_collected"], 1)
        self.assertIsNotNone(information)
        self.assertEqual(len(samples), 1)


if __name__ == "__main__":
    unittest.main()
