from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote
from finrl.rules.aggregation import share_weighted_average


def midpoint(quote: Quote) -> Decimal:
    return (quote.bid_price + quote.ask_price) / Decimal("2")


def quoted_spread(quote: Quote) -> Decimal:
    return quote.ask_price - quote.bid_price


def price_improvement(
    side: OrderSide,
    execution_price: Decimal,
    quote: Quote,
) -> Decimal:
    if side == OrderSide.BUY:
        return quote.ask_price - execution_price

    return execution_price - quote.bid_price


def effective_spread(
    side: OrderSide,
    execution_price: Decimal,
    quote: Quote,
) -> Decimal:
    return Decimal("2") * abs(
        execution_price - midpoint(quote)
    )


def share_weighted_price_improvement(
    side: OrderSide,
    executions: list[Execution],
    quote: Quote,
) -> Decimal | None:
    values = [
        (
            price_improvement(side, execution.price, quote),
            execution.quantity,
        )
        for execution in executions
    ]

    return share_weighted_average(values)


def share_weighted_effective_spread(
    side: OrderSide,
    executions: list[Execution],
    quote: Quote,
) -> Decimal | None:
    values = [
        (
            effective_spread(side, execution.price, quote),
            execution.quantity,
        )
        for execution in executions
    ]

    return share_weighted_average(values)


def realized_spread(
    side: OrderSide,
    execution_price: Decimal,
    future_quote: Quote,
) -> Decimal:
    future_midpoint = midpoint(future_quote)

    if side == OrderSide.BUY:
        return Decimal("2") * (execution_price - future_midpoint)

    return Decimal("2") * (future_midpoint - execution_price)


def share_weighted_realized_spread(
    side: OrderSide,
    executions: list[Execution],
    future_quotes: dict[str, Quote | None],
) -> Decimal | None:
    values = []

    for execution in executions:
        quote = future_quotes.get(execution.execution_id)

        if quote is None:
            return None

        values.append(
            (
                realized_spread(side, execution.price, quote),
                execution.quantity,
            )
        )

    return share_weighted_average(values)



