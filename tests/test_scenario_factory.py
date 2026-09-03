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
        execution_price=Decimal("100.00"),
        execution_quantity=Decimal("100"),
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
        execution_quantity=Decimal("60"),
        limit_price=Decimal("100.10"),
        bid_price=Decimal("99.90"),
        ask_price=Decimal("100.10"),
        execution_price=Decimal("100.00"),
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
        execution_quantity=Decimal("0"),
        limit_price=Decimal("100.10"),
        bid_price=Decimal("99.90"),
        ask_price=Decimal("100.10"),
        execution_price=Decimal("100.00"),
    )

    assert scenario["order"]["quantity"] == "100"
    assert scenario["executions"] == []