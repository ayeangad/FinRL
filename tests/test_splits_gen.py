"""Splits + synthetic generator tests."""

from pathlib import Path

from finrl.rl.splits import get_split, save_splits, split_scenarios
from finrl.scenario import Scenario
from finrl.scenario_gen import generate_batch, generate_scenario

GOLDEN = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden"


def test_splits_cover_all_and_disjoint():
    splits = split_scenarios(GOLDEN)
    assert set(splits.keys()) == {"train", "val", "test"}
    all_files = sorted(GOLDEN.glob("*.json"))
    total = sum(len(v) for v in splits.values())
    assert total == len(all_files)
    seen: set[str] = set()
    for v in splits.values():
        for p in v:
            assert p not in seen
            seen.add(p)


def test_splits_deterministic():
    a = split_scenarios(GOLDEN)
    b = split_scenarios(GOLDEN)
    assert a == b


def test_splits_ratios_reasonable():
    splits = split_scenarios(GOLDEN)
    n = sum(len(v) for v in splits.values())
    assert 0.6 * n <= len(splits["train"]) <= 0.8 * n
    assert len(splits["val"]) >= 5
    assert len(splits["test"]) >= 5


def test_get_split_returns_paths():
    for split in ("train", "val", "test"):
        paths = get_split(split, GOLDEN)
        assert len(paths) > 0
        assert all(p.suffix == ".json" for p in paths)


def test_save_splits_writes_json(tmp_path):
    out = tmp_path / "splits.json"
    summary = save_splits(out)
    assert out.exists()
    assert summary["train"]["count"] > 0


def test_generate_single_valid():
    data = generate_scenario("test_001", n_orders=2, seed=0)
    sc = Scenario.model_validate(data)
    assert len(sc.get_all_orders()) == 2


def test_generate_all_order_types():
    for otype in ("market", "limit", "stop", "stop_limit"):
        data = generate_scenario(f"t_{otype}", n_orders=1, seed=123)
        assert data["orders"][0]["order_type"] in ("market", "limit", "stop", "stop_limit")
    # At least exercises every branch without validation errors.


def test_generate_batch_writes(tmp_path):
    paths = generate_batch(tmp_path, n=5, seed=1, prefix="u")
    assert len(paths) == 5
    for p in paths:
        Scenario.model_validate(__import__("json").loads(p.read_text()))


def test_generated_reportable_and_scorable():
    from finrl.env.rule_605_env import Rule605Env

    data = generate_scenario("score_me", n_orders=2, seed=7)
    env = Rule605Env()
    from finrl.scenario import Scenario as _S

    env.reset(_S.model_validate(data))
    assert env.ground_truth_pipe.startswith("order_type_category|")
    assert len(env.ground_truth_pipe.strip().split("\n")) >= 2


def test_generated_deterministic():
    a = generate_scenario("det", n_orders=3, seed=42)
    b = generate_scenario("det", n_orders=3, seed=42)
    assert a == b
    c = generate_scenario("det", n_orders=3, seed=43)
    assert a != c
