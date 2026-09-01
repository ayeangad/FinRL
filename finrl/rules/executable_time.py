from datetime import datetime

from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType


def executable_time(
    order: Order,
    market: MarketState | None = None,
) -> datetime:
    if order.order_type in {
        OrderType.MARKET,
        OrderType.LIMIT,
    }:
        return order.received_at

    if order.order_type in {
        OrderType.STOP,
        OrderType.STOP_LIMIT,
    }:
        if market is None:
            raise ValueError(
                "Market state is required for stop orders"
            )

        if order.stop_price is None:
            raise ValueError(
                "Stop order requires stop_price"
            )

        for quote in sorted(
            market.quotes,
            key=lambda quote: quote.timestamp,
        ):
            if quote.timestamp < order.received_at:
                continue

            if order.side == OrderSide.BUY:
                triggered = (
                    quote.ask_price >= order.stop_price
                )
            else:
                triggered = (
                    quote.bid_price <= order.stop_price
                )

            if triggered:
                return quote.timestamp

        raise ValueError(
            "Stop order was not triggered"
        )

    raise ValueError(
        f"Unsupported order type: {order.order_type}"
    )
