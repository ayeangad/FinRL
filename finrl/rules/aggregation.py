from decimal import Decimal

from finrl.domain.execution import Execution


def total_executed_quantity(
    executions: list[Execution],
) -> Decimal:
    return sum(
        (execution.quantity for execution in executions),
        Decimal("0"),
    )


def volume_weighted_average_execution_price(
    executions: list[Execution],
) -> Decimal | None:
    if not executions:
        return None

    total_quantity = total_executed_quantity(executions)

    if total_quantity == 0:
        return None

    total_value = sum(
        (
            execution.price * execution.quantity
            for execution in executions
        ),
        Decimal("0"),
    )

    return total_value / total_quantity
