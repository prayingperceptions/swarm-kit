"""Worker identity: deterministic short handles for swarm workers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class WorkerIdentity:
    """An immutable identity for one swarm worker.

    ``handle`` is the canonical short name: ``worker-{index:04d}``.
    """

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be >= 0")

    @property
    def handle(self) -> str:
        """Short human-readable handle, e.g. ``worker-0007``."""
        return f"worker-{self.index:04d}"

    @classmethod
    def from_wallet(cls, address: str) -> "WorkerIdentity":
        """Derive a worker identity deterministically from a wallet address.

        index = int(sha256(address.lower().encode()).hexdigest(), 16) % 100000

        This is a stub for the wallet-derived identity pattern: real
        deployments derive worker identity from on-chain identity (e.g. a
        verified wallet binding) rather than a bare integer slot. The
        derivation here is deterministic but carries no on-chain authority —
        treat it as a namespace assignment, not authentication.
        """
        digest = sha256(address.lower().encode()).hexdigest()
        return cls(int(digest, 16) % 100000)
