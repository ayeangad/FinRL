from decimal import Decimal
from pathlib import Path

from finrl.scenarios import (
    load_executions,
    load_order,
    load_quotes,
    load_scenario,
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
