from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import app
from scripts.misbot_information import InformationRecord
from scripts.misbot_io import UserRecord


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_demo_runs_all_four_agents(self) -> None:
        response = self.client.get("/api/demo")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body["account"]["agent_scores"]), {"behavior_agent", "text_agent"})
        self.assertEqual(set(body["information"]["agent_scores"]), {"relation_agent", "propagation_agent"})

    def test_rejects_negative_information_counts(self) -> None:
        response = self.client.post(
            "/api/detect/information",
            json={"target_id": "x", "repost_count": -1},
        )
        self.assertEqual(response.status_code, 422)

    def test_rejects_invalid_weibo_target_before_network_access(self) -> None:
        response = self.client.post(
            "/api/detect/weibo",
            json={"target": "not-a-numeric-weibo-profile"},
        )
        self.assertEqual(response.status_code, 422)

    def test_weibo_request_returns_account_and_information_results(self) -> None:
        user = UserRecord("123", ["公开微博"], "简介", [10, 2, 5], [0] * 20, None)
        information = InformationRecord(
            "unknown", "公开微博", None, 3, 2, 8,
            ["456789"], [], [], 2, 1, 0, 0, [1.0],
            [("__source__", "456789", "comment")], [(1.0, "comment")],
        )
        profile = {
            "uid": "123",
            "screen_name": "测试账号",
            "followers_count": 10,
            "follow_count": 2,
            "statuses_count": 5,
            "posts_collected": 1,
            "latest_post": {"id": "post-1", "content": "公开微博"},
        }
        with patch(
            "backend.app.fetch_weibo_account",
            return_value=(user, information, profile, [(information, profile["latest_post"])]),
        ):
            response = self.client.post("/api/detect/weibo", json={"target": "123"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["account"]["target_type"], "account")
        self.assertEqual(body["information"]["target_type"], "information")
        self.assertEqual(body["interaction_graph"]["summary"]["post_count"], 1)
        self.assertEqual(body["interaction_graph"]["summary"]["edge_count"], 2)
        self.assertEqual(body["interaction_graph"]["nodes"][0]["type"], "account")
        self.assertEqual(body["interaction_graph"]["nodes"][2]["name"], "用户…6789")
        self.assertEqual(body["interaction_graph"]["links"][0]["action"], "publish")


if __name__ == "__main__":
    unittest.main()
