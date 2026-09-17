"""Eval determinism: same seed -> identical JSON (Hyde reproducibility)."""

import json


def test_eval_random_deterministic(tmp_path):
    from finrl.rl.evaluate import run_eval

    o1 = tmp_path / "a.json"
    o2 = tmp_path / "b.json"
    r1 = run_eval(env="tool", split="val", episodes=1, policies=["random"], max_steps=8, seed=0, out=o1)
    r2 = run_eval(env="tool", split="val", episodes=1, policies=["random"], max_steps=8, seed=0, out=o2)
    assert r1["policies"]["random"]["mean_dense"] == r2["policies"]["random"]["mean_dense"]
    assert r1["policies"]["random"]["per_scenario"] == r2["policies"]["random"]["per_scenario"]


def test_splits_json_matches_code():
    import json
    from pathlib import Path

    from finrl.rl.splits import split_scenarios

    p = Path(__file__).parent.parent / "experiments" / "rl" / "splits.json"
    assert p.exists()
    saved = json.loads(p.read_text())
    live = split_scenarios()
    assert saved["train"]["count"] == len(live["train"])
    assert saved["val"]["count"] == len(live["val"])
    assert saved["test"]["count"] == len(live["test"])
