from decimal import Decimal

from finrl.domain.order import Order, OrderSide, OrderType
from finrl.rules.classification import (
    OrderClass,
    classify_order,
)


def make_order(side: OrderSide, order_type: OrderType, limit_price=None) -> Order:
    return Order(
        order_id="ORD-0001",
        security="FINRL",
        side=side,
        order_type=order_type,
        quantity=Decimal("100"),
        limit_price=limit_price,
        received_at="2026-09-01T10:30:00.000",
    )


def test_market_order():
    order = make_order(
        OrderSide.BUY,
        OrderType.MARKET,
    )

    assert classify_order(
        order,
        Decimal("99.90"),
        Decimal("100.10"),
    ) == OrderClass.MARKET


def test_buy_limit_at_ask_is_marketable():
    order = make_order(
        OrderSide.BUY,
        OrderType.LIMIT,
        Decimal("100.10"),
    )

    assert classify_order(
        order,
        Decimal("99.90"),
        Decimal("100.10"),
    ) == OrderClass.MARKETABLE_LIMIT


def test_buy_limit_above_ask_is_marketable():
    order = make_order(
        OrderSide.BUY,
        OrderType.LIMIT,
        Decimal("100.20"),
    )

    assert classify_order(
        order,
        Decimal("99.90"),
        Decimal("100.10"),
    ) == OrderClass.MARKETABLE_LIMIT


def test_buy_limit_below_ask_is_non_marketable():
    order = make_order(
        OrderSide.BUY,
        OrderType.LIMIT,
        Decimal("100.00"),
    )

    assert classify_order(
        order,
        Decimal("99.90"),
        Decimal("100.10"),
    ) == OrderClass.NON_MARKETABLE_LIMIT


def test_sell_limit_at_bid_is_marketable():
    order = make_order(
        OrderSide.SELL,
        OrderType.LIMIT,
        Decimal("99.90"),
    )

    assert classify_order(
        order,
        Decimal("99.90"),
        Decimal("100.10"),
    ) == OrderClass.MARKETABLE_LIMIT


def test_sell_limit_below_bid_is_marketable():
    order = make_order(
        OrderSide.SELL,
        OrderType.LIMIT,
        Decimal("99.80"),
    )

    assert classify_order(
        order,
        Decimal("99.90"),
        Decimal("100.10"),
    ) == OrderClass.MARKETABLE_LIMIT


def test_sell_limit_above_bid_is_non_marketable():
    order = make_order(
        OrderSide.SELL,
        OrderType.LIMIT,
        Decimal("100.00"),
    )

    assert classify_order(
        order,
        Decimal("99.90"),
        Decimal("100.10"),
    ) == OrderClass.NON_MARKETABLE_LIMIT


def test_stop_and_stop_limit_orders_raise_value_error():
    stop_order = Order(
        order_id="ORD-STOP",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        quantity=Decimal("100"),
        stop_price=Decimal("100.00"),
        received_at="2026-09-01T10:30:00.000",
    )

    stop_limit_order = Order(
        order_id="ORD-STOP-LIMIT",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        quantity=Decimal("100"),
        stop_price=Decimal("100.00"),
        limit_price=Decimal("100.10"),
        received_at="2026-09-01T10:30:00.000",
    )

    import pytest

    with pytest.raises(ValueError, match="Unsupported order type"):
        classify_order(stop_order, Decimal("99.90"), Decimal("100.10"))

    with pytest.raises(ValueError, match="Unsupported order type"):
        classify_order(stop_limit_order, Decimal("99.90"), Decimal("100.10"))


def test_limit_order_without_limit_price_raises_value_error():
    order = Order(
        order_id="ORD-NO-LIMIT",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=None,
        received_at="2026-09-01T10:30:00.000",
    )

    import pytest

    with pytest.raises(ValueError, match="Limit order requires limit_price"):
        classify_order(order, Decimal("99.90"), Decimal("100.10"))

