from decimal import Decimal
import pytest
from finrl.rules.order_result import OrderExecutionResult


def test_fully_executed_order():
    result = OrderExecutionResult(
    order_id="ORD-0001",
    requested_quantity=Decimal("100"),
    executed_quantity=Decimal("100"),
    average_execution_price=Decimal("100.12"),
)

    assert result.executed_quantity == Decimal("100")
    assert result.fully_executed is True
    assert result.partially_executed is False
    assert result.unexecuted is False


def test_partially_executed_order():
    result = OrderExecutionResult(
        order_id="ORD-0002",
        requested_quantity=Decimal("100"),
        executed_quantity=Decimal("40"),
        average_execution_price=Decimal("100.05"),
        fully_executed=False,
        partially_executed=True,
        unexecuted=False,
    )

    assert result.executed_quantity == Decimal("40")
    assert result.partially_executed is True
    assert result.fully_executed is False


def test_unexecuted_order():
    result = OrderExecutionResult(
        order_id="ORD-0003",
        requested_quantity=Decimal("100"),
        executed_quantity=Decimal("0"),
        average_execution_price=None,
        fully_executed=False,
        partially_executed=False,
        unexecuted=True,
    )

    assert result.executed_quantity == Decimal("0")
    assert result.average_execution_price is None
    assert result.unexecuted is True

def test_executed_quantity_cannot_exceed_requested_quantity():
    with pytest.raises(ValueError):
        OrderExecutionResult(
            order_id="ORD-0004",
            requested_quantity=Decimal("100"),
            executed_quantity=Decimal("101"),
            average_execution_price=Decimal("100.00"),
        )
    