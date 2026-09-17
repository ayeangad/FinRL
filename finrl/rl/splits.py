"""Deterministic train/val/test splits over golden scenarios.

Split is hash-stable (sha256 of scenario stem + salt) so train/val/test
membership never shifts when files are added. Saved to
experiments/rl/splits.json for auditability.

Default 70/15/15 over scenarios/v0.1/golden/*.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_DIR = Path("scenarios/v0.1/golden")
SPLITS_PATH = Path("experiments/rl/splits.json")


def _hash_fraction(key: str, salt: str = "finrl-v1") -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_scenarios(
    scenarios_dir: str | Path = DEFAULT_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    salt: str = "finrl-v1",
) -> dict[str, list[str]]:
    files = sorted(Path(scenarios_dir).glob("*.json"))
    train, val, test = [], [], []
    for f in files:
        x = _hash_fraction(f.stem, salt)
        if x < train_ratio:
            train.append(str(f))
        elif x < train_ratio + val_ratio:
            val.append(str(f))
        else:
            test.append(str(f))
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def get_split(
    split: str,
    scenarios_dir: str | Path = DEFAULT_DIR,
    **kwargs,
) -> list[Path]:
    splits = split_scenarios(scenarios_dir, **kwargs)
    if split not in splits:
        raise ValueError(f"split must be one of {sorted(splits)}, got {split!r}")
    return [Path(p) for p in splits[split]]


def save_splits(path: str | Path = SPLITS_PATH, **kwargs) -> dict:
    splits = split_scenarios(**kwargs)
    summary = {
        k: {"count": len(v), "files": [Path(p).name for p in v]}
        for k, v in splits.items()
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2))
    return summary
