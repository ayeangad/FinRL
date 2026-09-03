from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderType
from finrl.rules.executable_time import executable_time


def is_reportable_order(
    order: Order,
    executions: list[Execution],
    market: MarketState,
) -> bool:
    receipt_quote = market.quote_at(order.received_at)
    if receipt_quote is None:
        return False

    if order.order_type == OrderType.MARKET:
        return True

    if order.order_type in {
        OrderType.LIMIT,
        OrderType.STOP,
        OrderType.STOP_LIMIT,
    }:
        try:
            executable_time(order, market)
            return True
        except ValueError:
            return False

    return False
