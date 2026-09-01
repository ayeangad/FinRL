from decimal import Decimal
from pathlib import Path

from finrl.domain.market import MarketState
from finrl.evals.order_evaluator import evaluate_order
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


def test_basic_limit_order_scenario_evaluation():
    scenario = load_scenario(SCENARIO_PATH)

    order = load_order(
        scenario["order"],
        scenario["security"],
    )
    quotes = load_quotes(scenario["quotes"])
    executions = load_executions(scenario["executions"])

    market = MarketState(
        security=scenario["security"],
        quotes=quotes,
    )

    report = evaluate_order(
        order=order,
        executions=executions,
        market=market,
    )

    expected = scenario["expected"]

    assert report.order_id == scenario["order"]["order_id"]
    assert report.executed_quantity == Decimal(expected["executed_quantity"])
    assert report.average_execution_price == Decimal(
        expected["average_execution_price"]
    )
    assert report.quoted_spread == Decimal(expected["quoted_spread"])
    assert report.price_improvement == Decimal(expected["price_improvement"])
    assert report.effective_spread == Decimal(expected["effective_spread"])

