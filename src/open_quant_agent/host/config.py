from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PermissionPolicy:
    profile: str = "research_safe"
    research_only: bool = True
    no_network_default: bool = True
    allow_broker_integrations: bool = False
    allow_generated_code_execution: bool = False


@dataclass
class HostConfig:
    sessions_root: Path = Path("runs/sessions")
    memory_root: Path = Path("memory")
    registry_root: Path = Path("registries")
    default_state_dir: Path = Path(".agent_state")
    permission_policy: PermissionPolicy | None = None

    def __post_init__(self) -> None:
        self.sessions_root = self.sessions_root.expanduser().resolve()
        self.memory_root = self.memory_root.expanduser().resolve()
        self.registry_root = self.registry_root.expanduser().resolve()
        self.default_state_dir = self.default_state_dir.expanduser()
        if self.permission_policy is None:
            self.permission_policy = PermissionPolicy()
