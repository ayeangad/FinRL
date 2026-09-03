from datetime import datetime

from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType


def executable_time(
    order: Order,
    market: MarketState | None = None,
) -> datetime:
    if order.order_type == OrderType.MARKET:
        return order.received_at

    if order.order_type == OrderType.LIMIT:
        if order.limit_price is None:
            raise ValueError("Limit order requires limit_price")

        if market is None:
            return order.received_at

        receipt_quote = market.quote_at(order.received_at)
        if receipt_quote is not None:
            if (
                order.side == OrderSide.BUY
                and order.limit_price >= receipt_quote.ask_price
            ):
                return order.received_at
            if (
                order.side == OrderSide.SELL
                and order.limit_price <= receipt_quote.bid_price
            ):
                return order.received_at

        for quote in sorted(market.quotes, key=lambda q: q.timestamp):
            if quote.timestamp < order.received_at:
                continue

            if order.side == OrderSide.BUY:
                executable = order.limit_price >= quote.bid_price
            else:
                executable = order.limit_price <= quote.ask_price

            if executable:
                return quote.timestamp

        raise ValueError("Limit order was not executable")

    if order.order_type == OrderType.STOP:
        if market is None:
            raise ValueError("Market state is required for stop orders")

        if order.stop_price is None:
            raise ValueError("Stop order requires stop_price")

        for quote in sorted(market.quotes, key=lambda q: q.timestamp):
            if quote.timestamp < order.received_at:
                continue

            if order.side == OrderSide.BUY:
                triggered = quote.ask_price >= order.stop_price
            else:
                triggered = quote.bid_price <= order.stop_price

            if triggered:
                return quote.timestamp

        raise ValueError("Stop order was not triggered")

    if order.order_type == OrderType.STOP_LIMIT:
        if market is None:
            raise ValueError("Market state is required for stop orders")

        if order.stop_price is None:
            raise ValueError("Stop order requires stop_price")

        if order.limit_price is None:
            raise ValueError("Stop-limit order requires limit_price")

        trigger_time: datetime | None = None
        for quote in sorted(market.quotes, key=lambda q: q.timestamp):
            if quote.timestamp < order.received_at:
                continue

            if trigger_time is None:
                if order.side == OrderSide.BUY:
                    triggered = quote.ask_price >= order.stop_price
                else:
                    triggered = quote.bid_price <= order.stop_price

                if triggered:
                    trigger_time = quote.timestamp

            if trigger_time is not None:
                if order.side == OrderSide.BUY:
                    executable = order.limit_price >= quote.bid_price
                else:
                    executable = order.limit_price <= quote.ask_price

                if executable:
                    return quote.timestamp

        if trigger_time is None:
            raise ValueError("Stop-limit order was not triggered")

        raise ValueError("Stop-limit order limit was not executable")


    raise ValueError(f"Unsupported order type: {order.order_type}")

