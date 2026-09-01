from enum import Enum
from decimal import Decimal

from finrl.domain.order import Order, OrderType


class OrderClass(str, Enum):
    MARKET = "market"
    MARKETABLE_LIMIT = "marketable_limit"
    NON_MARKETABLE_LIMIT = "non_marketable_limit"


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
