from __future__ import annotations

import json

from agents import BotArenaRuntime
from scripts.misbot_information import iter_information
from scripts.misbot_io import MisBotDataset


def main() -> None:
    dataset = MisBotDataset()
    runtime = BotArenaRuntime()
    user = next(dataset.iter_users(sampled=True, limit=1))
    information = next(iter_information(dataset, "misinformation", limit=1))
    output = {
        "account_detection": runtime.analyze_user(user),
        "information_detection": runtime.analyze_information(information, "misinformation_0000"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

