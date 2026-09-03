from datetime import datetime
from decimal import Decimal

from finrl.rules.classification import OrderTypeCategory
from finrl.rules.order_size import OrderSizeBucket
from finrl.scenario import Scenario
from finrl.scenario_runner import load_scenario, run_scenario, run_scenario_and_serialize


def test_scenario_runner_basic_limit_order():
    scenario_data = {
        "scenario_id": "test_001",
        "version": "v0.1",
        "security": "FINRL",
        "order": {
            "order_id": "ORD-0001",
            "side": "buy",
            "order_type": "limit",
            "quantity": "100",
            "limit_price": "100.10",
            "received_at": "2026-09-01T10:30:00.100",
        },
        "quotes": [
            {
                "security": "FINRL",
                "bid_price": "99.90",
                "bid_size": "500",
                "ask_price": "100.10",
                "ask_size": "300",
                "timestamp": "2026-09-01T10:30:00.100",
            }
        ],
        "executions": [
            {
                "execution_id": "EXE-0001",
                "order_id": "ORD-0001",
                "price": "100.00",
                "quantity": "100",
                "executed_at": "2026-09-01T10:30:00.200",
            }
        ],
    }

    sc = load_scenario(scenario_data)
    report = run_scenario(sc)

    cell = report.get_cell(
        OrderTypeCategory.MARKETABLE_LIMIT, OrderSizeBucket.SHARES_100_TO_499
    )
    assert cell.order_count == 1
    assert cell.executed_order_count == 1
    assert cell.total_executed_quantity == Decimal("100")
    assert cell.price_improvement == Decimal("0.10")
    assert cell.effective_spread == Decimal("0.00")
    assert cell.quoted_spread == Decimal("0.20")

    pipe_output = run_scenario_and_serialize(sc, format="pipe")
    assert "marketable_limit|100_to_499|1|1|100|100|100|0|0|0.10|0.00|0.20" in pipe_output
