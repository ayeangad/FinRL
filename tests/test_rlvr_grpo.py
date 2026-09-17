"""RLVR dataset + GRPO reward-seam tests."""

import json


def test_export_train_split_only(tmp_path):
    from finrl.rl.dataset import export_rlvr_dataset
    from finrl.rl.splits import get_split

    out = tmp_path / "rlvr.jsonl"
    res = export_rlvr_dataset(out_path=out, split="train", limit=5)
    assert res["items"] == 5
    assert res["split"] == "train"
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert len(rows) == 5
    train_stems = {p.stem for p in get_split("train")}
    for r in rows:
        assert r["scenario_id"] in train_stems
        assert r["dataset_version"].startswith("rlvr-")
        # No GT leak into prompt.
        assert "order_type_category|" not in r["prompt"]
        assert r["ground_truth_pipe"].startswith("order_type_category|")


def test_grpo_smoke(tmp_path):
    from finrl.rl.dataset import export_rlvr_dataset
    from finrl.rl.train_grpo_stub import reward_for_pair, smoke

    ds = tmp_path / "d.jsonl"
    export_rlvr_dataset(out_path=ds, split="train", limit=5)
    res = smoke(ds, limit=5)
    assert res["ref_mean"] == 1.0
    assert res["broken_mean"] == 0.0


def test_reward_fn_is_dense():
    from finrl.rl.train_grpo_stub import reward_for_pair

    out = reward_for_pair("bad", "order_type_category|x\nm|1")
    assert 0.0 <= out["reward"] <= 1.0
    assert "regulatory_accuracy" in out
