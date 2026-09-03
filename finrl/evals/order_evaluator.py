from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order
from finrl.rules.aggregation import (
    total_executed_quantity,
    volume_weighted_average_execution_price,
)
from finrl.rules.execution_context import realized_spread_at_horizon
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.metrics import (
    effective_spread,
    price_improvement,
    quoted_spread,
    share_weighted_effective_spread,
    share_weighted_price_improvement,
)
from finrl.rules.order_result import OrderExecutionResult
from finrl.rules.report import OrderReport


def evaluate_order(
    order: Order,
    executions: list[Execution],
    market: MarketState,
) -> OrderReport:
    executed_quantity = total_executed_quantity(executions)
    average_price = volume_weighted_average_execution_price(executions)

    execution_result = OrderExecutionResult(
        order_id=order.order_id,
        requested_quantity=order.quantity,
        executed_quantity=executed_quantity,
        average_execution_price=average_price,
    )

    if average_price is None:
        return OrderReport(
            order_id=order.order_id,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            average_execution_price=None,
            price_improvement=None,
            effective_spread=None,
            quoted_spread=None,
            realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
        )

    quote = market.quote_at(order.received_at)

    if quote is None:
        return OrderReport(
            order_id=order.order_id,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            average_execution_price=average_price,
            price_improvement=None,
            effective_spread=None,
            quoted_spread=None,
            realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
        )

    realized_spreads = {
        horizon: realized_spread_at_horizon(
            order.side,
            executions,
            market,
            horizon.duration,
        )
        for horizon in RealizedSpreadHorizon
    }

    return OrderReport(
        order_id=order.order_id,
        requested_quantity=order.quantity,
        executed_quantity=execution_result.executed_quantity,
        average_execution_price=execution_result.average_execution_price,
        price_improvement=share_weighted_price_improvement(
            order.side,
            executions,
            quote,
        ),
        effective_spread=share_weighted_effective_spread(
            order.side,
            executions,
            quote,
        ),
        quoted_spread=quoted_spread(quote),
        realized_spreads=realized_spreads,
    )

