from decimal import Decimal
from enum import Enum

from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote


class OrderClass(str, Enum):
    MARKET = "market"
    MARKETABLE_LIMIT = "marketable_limit"
    NON_MARKETABLE_LIMIT = "non_marketable_limit"


class OrderTypeCategory(str, Enum):
    MARKET = "market"
    MARKETABLE_LIMIT = "marketable_limit"
    NON_MARKETABLE_LIMIT = "non_marketable_limit"
    MIDPOINT_OR_BETTER_LIMIT = "midpoint_or_better_limit"
    STOP = "stop"


def classify_order(
    order: Order,
    bid_price: Decimal,
    ask_price: Decimal,
) -> OrderClass:
    if order.order_type == OrderType.MARKET:
        return OrderClass.MARKET

    if order.order_type != OrderType.LIMIT:
        raise ValueError(
            f"Unsupported order type: {order.order_type}"
        )

    if order.limit_price is None:
        raise ValueError("Limit order requires limit_price")

    if order.side.value == "buy":
        if order.limit_price >= ask_price:
            return OrderClass.MARKETABLE_LIMIT

        return OrderClass.NON_MARKETABLE_LIMIT

    if order.limit_price <= bid_price:
        return OrderClass.MARKETABLE_LIMIT

    return OrderClass.NON_MARKETABLE_LIMIT


def classify_order_category(
    order: Order,
    receipt_quote: Quote | None,
) -> OrderTypeCategory:
    if order.order_type == OrderType.MARKET:
        return OrderTypeCategory.MARKET

    if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
        return OrderTypeCategory.STOP

    if order.order_type != OrderType.LIMIT:
        raise ValueError(
            f"Unsupported order type for category classification: {order.order_type}"
        )

    if order.limit_price is None:
        raise ValueError("Limit order requires limit_price")

    if receipt_quote is None:
        raise ValueError("Receipt quote is required to classify limit order category")

    bid_price = receipt_quote.bid_price
    ask_price = receipt_quote.ask_price
    midpoint = (bid_price + ask_price) / Decimal("2")

    if order.side == OrderSide.BUY:
        if order.limit_price >= ask_price:
            return OrderTypeCategory.MARKETABLE_LIMIT
        if order.limit_price >= midpoint:
            return OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT
        return OrderTypeCategory.NON_MARKETABLE_LIMIT

    if order.limit_price <= bid_price:
        return OrderTypeCategory.MARKETABLE_LIMIT

    if order.limit_price <= midpoint:
        return OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT

    return OrderTypeCategory.NON_MARKETABLE_LIMIT

