from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from open_quant_agent import MultiAgentQuantOrchestrator, OrchestratorConfig


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        config = OrchestratorConfig(
            max_iterations=1,
            offline_synthetic=True,
            universe_size=16,
            state_dir=Path(tmpdir) / ".agent_state",
        )
        records = MultiAgentQuantOrchestrator(config).run()
    latest = records[-1] if records else {}
    print(
        json.dumps(
            {
                "cycle": latest.get("cycle"),
                "best_factor": (latest.get("best") or {}).get("factor_name"),
                "best_factor_status": (latest.get("best") or {}).get("status"),
                "best_strategy": (latest.get("best_strategy") or {}).get("strategy_name"),
                "best_strategy_status": (latest.get("best_strategy") or {}).get("status"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
