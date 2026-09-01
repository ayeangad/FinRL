from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.rules.aggregation import (
    total_executed_quantity,
    volume_weighted_average_execution_price,
)


def make_execution(
    execution_id: str,
    quantity: str,
    price: str,
) -> Execution:
    return Execution(
        execution_id=execution_id,
        order_id="ORD-0001",
        quantity=Decimal(quantity),
        price=Decimal(price),
        executed_at="2026-09-01T10:30:00.000",
    )


def test_total_executed_quantity():
    executions = [
        make_execution("EXE-0001", "40", "100.00"),
        make_execution("EXE-0002", "60", "100.20"),
    ]

    assert total_executed_quantity(executions) == Decimal("100")


def test_volume_weighted_average_execution_price():
    executions = [
        make_execution("EXE-0001", "40", "100.00"),
        make_execution("EXE-0002", "60", "100.20"),
    ]

    assert volume_weighted_average_execution_price(
        executions
    ) == Decimal("100.12")


def test_empty_executions_have_zero_quantity():
    assert total_executed_quantity([]) == Decimal("0")


def test_empty_executions_have_no_average_price():
    assert volume_weighted_average_execution_price([]) is None