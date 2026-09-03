from datetime import datetime
from decimal import Decimal

from finrl.domain.order import OrderSide, OrderType
from finrl.scenario_factory import make_limit_order_scenario


def test_make_limit_order_scenario():
    scenario = make_limit_order_scenario(
        scenario_id="factory_test_001",
        security="FINRL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        limit_price=Decimal("100.10"),
        bid_price=Decimal("99.90"),
        ask_price=Decimal("100.10"),
        executions=[
            (
                Decimal("100.00"),
                Decimal("100"),
                datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
        ],
    )

    assert scenario["scenario_id"] == "factory_test_001"
    assert scenario["version"] == "v0.1"
    assert scenario["security"] == "FINRL"

    assert scenario["order"]["side"] == "buy"
    assert scenario["order"]["order_type"] == OrderType.LIMIT.value
    assert scenario["order"]["quantity"] == "100"
    assert scenario["order"]["limit_price"] == "100.10"

    assert scenario["quotes"][0]["bid_price"] == "99.90"
    assert scenario["quotes"][0]["ask_price"] == "100.10"

    assert scenario["executions"][0]["quantity"] == "100"
    assert scenario["executions"][0]["price"] == "100.00"

    assert scenario["expected"] == {
        "executed_quantity": "100",
        "average_execution_price": "100.00",
        "quoted_spread": "0.20",
        "price_improvement": "0.10",
        "effective_spread": "0.00",
    }

def test_make_limit_order_scenario_supports_partial_execution():
    scenario = make_limit_order_scenario(
        scenario_id="factory_partial_001",
        security="FINRL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        executions=[
            (
                Decimal("100.00"),
                Decimal("60"),
                datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
        ],
        limit_price=Decimal("100.10"),
        bid_price=Decimal("99.90"),
        ask_price=Decimal("100.10"),
    )

    assert scenario["order"]["quantity"] == "100"
    assert len(scenario["executions"]) == 1
    assert scenario["executions"][0]["quantity"] == "60"

def test_make_limit_order_scenario_supports_unexecuted_order():
    scenario = make_limit_order_scenario(
        scenario_id="factory_unexecuted_001",
        security="FINRL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        executions=[],
        limit_price=Decimal("100.10"),
        bid_price=Decimal("99.90"),
        ask_price=Decimal("100.10"),
    )
    
    assert scenario["order"]["quantity"] == "100"
    assert scenario["executions"] == []

def test_make_limit_order_scenario_supports_multiple_executions():
    scenario = make_limit_order_scenario(
        scenario_id="factory_multifill_001",
        security="FINRL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        executions=[
            (
                Decimal("100.00"),
                Decimal("30"),
                datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
            (
                Decimal("100.02"),
                Decimal("40"),
                datetime.fromisoformat("2026-09-01T10:30:00.300"),
            ),
            (
                Decimal("99.98"),
                Decimal("30"),
                datetime.fromisoformat("2026-09-01T10:30:00.400"),
            ),
        ],
        limit_price=Decimal("100.10"),
        bid_price=Decimal("99.90"),
        ask_price=Decimal("100.10"),
    )

    assert len(scenario["executions"]) == 3

    assert scenario["executions"][0]["quantity"] == "30"
    assert scenario["executions"][1]["quantity"] == "40"
    assert scenario["executions"][2]["quantity"] == "30"

    assert scenario["executions"][0]["price"] == "100.00"
    assert scenario["executions"][1]["price"] == "100.02"
    assert scenario["executions"][2]["price"] == "99.98"
    assert scenario["expected"]["executed_quantity"] == "100"
    assert scenario["expected"]["average_execution_price"] == "100.002"

    assert scenario["executions"][0]["executed_at"] == "2026-09-01T10:30:00.200000"
    assert scenario["executions"][1]["executed_at"] == "2026-09-01T10:30:00.300000"
    assert scenario["executions"][2]["executed_at"] == "2026-09-01T10:30:00.400000"