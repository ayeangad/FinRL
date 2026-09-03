from pathlib import Path
import pytest

from finrl.scenario_runner import load_scenario, run_scenario_and_serialize

GOLDEN_DIR = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden"
SCENARIO_FILES = sorted(list(GOLDEN_DIR.glob("*.json")))


def test_golden_scenarios_count():
    assert len(SCENARIO_FILES) == 100, f"Expected 100 golden scenarios, found {len(SCENARIO_FILES)}"


@pytest.mark.parametrize("scenario_path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_golden_scenario(scenario_path: Path):
    pipe_path = scenario_path.with_suffix(".pipe")
    assert pipe_path.exists(), f"Golden expected file missing for scenario {scenario_path.name}"

    scenario = load_scenario(scenario_path)
    actual_pipe = run_scenario_and_serialize(scenario, format="pipe")
    expected_pipe = pipe_path.read_text()

    assert actual_pipe == expected_pipe, f"Mismatch in golden scenario output for {scenario_path.name}"
