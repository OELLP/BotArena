from __future__ import annotations

import unittest

from agents.neural_models import graph_arrays, temporal_array
from scripts.misbot_information import InformationRecord, extract_propagation_features, extract_relation_features


class InformationRecordTests(unittest.TestCase):
    def test_participants_are_deduplicated_across_actions(self) -> None:
        record = InformationRecord(
            category="misinformation",
            content="示例",
            publish_time=1.0,
            repost_count=1,
            comment_count=1,
            attitude_count=1,
            comment_users=["u1", "u2"],
            repost_users=["u2", "u3"],
            attitude_users=["u1"],
            comment_nodes=2,
            comment_edges=1,
            repost_nodes=2,
            repost_edges=1,
            activity_times=[1.0, 30.0, 400.0],
            interaction_edges=[("u1", "u2", "comment"), ("u2", "u3", "repost")],
            activity_events=[(1.0, "comment"), (30.0, "comment"), (400.0, "repost")],
        )
        self.assertEqual(record.participants, {"u1", "u2", "u3"})
        self.assertEqual(len(extract_relation_features(record)), 9)
        self.assertEqual(len(extract_propagation_features(record)), 10)
        nodes, edges = graph_arrays(record)
        self.assertEqual(nodes.shape[1], 6)
        self.assertEqual(edges.shape[0], 2)
        self.assertEqual(temporal_array(record).shape, (3, 6))



if __name__ == "__main__":
    unittest.main()
