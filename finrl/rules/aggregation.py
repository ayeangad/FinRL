from decimal import Decimal

from finrl.domain.execution import Execution


def total_executed_quantity(
    executions: list[Execution],
) -> Decimal:
    return sum(
        (execution.quantity for execution in executions),
        Decimal(0),
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
        Decimal(0),
    )

    return total_value / total_quantity


def share_weighted_average(
    values: list[tuple[Decimal, Decimal]],
) -> Decimal | None:
    if not values:
        return None

    total_quantity = sum(
        (quantity for _, quantity in values),
        Decimal(0),
    )

    if total_quantity == 0:
        return None

    total_weighted_value = sum(
        (
            value * quantity
            for value, quantity in values
        ),
        Decimal(0),
    )

    return total_weighted_value / total_quantity

