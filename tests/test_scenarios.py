from decimal import Decimal
from pathlib import Path
import pytest
import json

from finrl.scenarios import (
    load_executions,
    load_market,
    load_order,
    load_quotes,
    load_scenario,
    validate_scenario,
)


SCENARIO_PATH = (
    Path(__file__).parent.parent
    / "scenarios"
    / "v0.1"
    / "basic_limit_order.json"
)


def test_load_scenario():
    scenario = load_scenario(SCENARIO_PATH)

    assert scenario["scenario_id"] == "basic_limit_order_001"
    assert scenario["version"] == "v0.1"
    assert scenario["security"] == "FINRL"


def test_load_order():
    scenario = load_scenario(SCENARIO_PATH)

    order = load_order(
        scenario["order"],
        scenario["security"],
    )

    assert order.order_id == "ORD-0001"
    assert order.security == "FINRL"
    assert order.quantity == Decimal("100")
    assert order.limit_price == Decimal("100.10")


def test_load_quotes():
    scenario = load_scenario(SCENARIO_PATH)
    quotes = load_quotes(scenario["quotes"])

    assert len(quotes) == 1
    assert quotes[0].bid_price == Decimal("99.90")
    assert quotes[0].ask_price == Decimal("100.10")


def test_load_executions():
    scenario = load_scenario(SCENARIO_PATH)
    executions = load_executions(scenario["executions"])

    assert len(executions) == 1
    assert executions[0].quantity == Decimal("100")
    assert executions[0].price == Decimal("100.00")


def test_validate_scenario_accepts_valid_scenario():
    scenario = load_scenario(SCENARIO_PATH)

    validate_scenario(scenario)

def test_validate_scenario_rejects_wrong_execution_order():
    scenario = load_scenario(SCENARIO_PATH)

    scenario["executions"][0]["order_id"] = "WRONG-ORDER"

    with pytest.raises(ValueError, match="references order"):
        validate_scenario(scenario)



def test_validate_scenario_rejects_excess_execution_quantity():
    scenario = load_scenario(SCENARIO_PATH)

    scenario["executions"][0]["quantity"] = "101"

    with pytest.raises(
        ValueError,
        match="Total executed quantity cannot exceed requested quantity",
    ):
        validate_scenario(scenario)


def test_validate_scenario_rejects_wrong_quote_security():
    scenario = load_scenario(SCENARIO_PATH)

    scenario["quotes"][0]["security"] = "OTHER-SECURITY"

    with pytest.raises(
        ValueError,
        match="Quote security .* does not match scenario security",
    ):
        validate_scenario(scenario)

def test_validate_scenario_accepts_valid_scenario():
    scenario = load_scenario(SCENARIO_PATH)

    validate_scenario(scenario)


def test_validate_scenario_rejects_wrong_execution_order():
    scenario = load_scenario(SCENARIO_PATH)

    scenario["executions"][0]["order_id"] = "WRONG-ORDER"

    with pytest.raises(ValueError, match="references order"):
        validate_scenario(scenario)


def test_validate_scenario_accepts_execution_within_order_quantity():
    scenario = load_scenario(SCENARIO_PATH)

    validate_scenario(scenario)


def test_validate_scenario_rejects_excess_execution_quantity():
    scenario = load_scenario(SCENARIO_PATH)

    scenario["executions"][0]["quantity"] = "101"

    with pytest.raises(
        ValueError,
        match="Total executed quantity cannot exceed requested quantity",
    ):
        validate_scenario(scenario)


def test_validate_scenario_accepts_matching_quote_security():
    scenario = load_scenario(SCENARIO_PATH)

    validate_scenario(scenario)


def test_validate_scenario_rejects_wrong_quote_security():
    scenario = load_scenario(SCENARIO_PATH)

    scenario["quotes"][0]["security"] = "OTHER-SECURITY"

    with pytest.raises(
        ValueError,
        match="Quote security .* does not match scenario security",
    ):
        validate_scenario(scenario)

def test_load_scenario_rejects_invalid_execution_order():
    scenario = load_scenario(SCENARIO_PATH)
    scenario["executions"][0]["order_id"] = "WRONG-ORDER"

    with pytest.raises(ValueError, match="references order"):
        validate_scenario(scenario)

def test_load_scenario_rejects_invalid_file(tmp_path):
    scenario = load_scenario(SCENARIO_PATH)
    scenario["executions"][0]["order_id"] = "WRONG-ORDER"

    invalid_path = tmp_path / "invalid_scenario.json"
    invalid_path.write_text(
        json.dumps(scenario),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="references order"):
        load_scenario(invalid_path)

def test_validate_scenario_rejects_missing_required_field():
    scenario = load_scenario(SCENARIO_PATH)
    del scenario["expected"]

    with pytest.raises(
        ValueError,
        match="Scenario is missing required fields",
    ):
        validate_scenario(scenario)


def test_validate_scenario_reports_all_missing_required_fields():
    scenario = load_scenario(SCENARIO_PATH)

    del scenario["expected"]
    del scenario["executions"]

    with pytest.raises(
        ValueError,
        match="Scenario is missing required fields",
    ) as exc_info:
        validate_scenario(scenario)

    message = str(exc_info.value)

    assert "expected" in message
    assert "executions" in message