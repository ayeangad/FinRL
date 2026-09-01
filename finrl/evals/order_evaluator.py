from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order
from finrl.rules.aggregation import (
    total_executed_quantity,
    volume_weighted_average_execution_price,
)

from finrl.rules.metrics import (
    effective_spread,
    price_improvement,
    quoted_spread,
)
from finrl.rules.order_result import OrderExecutionResult
from finrl.rules.report import OrderReport


def evaluate_order(
    order: Order,
    executions: list[Execution],
    market: MarketState
) -> OrderReport:
    executed_quantity = total_executed_quantity(executions)
    average_price = (volume_weighted_average_execution_price(executions))

    execution_result = OrderExecutionResult(
        order_id=order.order_id,
        requested_quantity=order.quantity,
        executed_quantity=executed_quantity,
        average_execution_price=average_price
    )

    quote = market.quote_at(order.received_at)

    if quote is None or average_price is None:
        return OrderReport(
            order_id=order.order_id,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            average_execution_price=average_price,
        )

    return OrderReport(
        order_id=order.order_id,
        requested_quantity=order.quantity,
        executed_quantity=execution_result.executed_quantity,
        average_execution_price=execution_result.average_execution_price,
        price_improvement=price_improvement(
            order.side,
            average_price,
            quote,
        ),
        effective_spread=effective_spread(
            order.side,
            average_price,
            quote,
        ),
        quoted_spread=quoted_spread(
            quote
        ),
        
    )

    