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



def test_classify_order_category_market_and_stop():
    from finrl.domain.quote import Quote
    from finrl.rules.classification import OrderTypeCategory, classify_order_category

    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.000",
    )

    mkt = make_order(OrderSide.BUY, OrderType.MARKET)
    stop = Order(
        order_id="S1",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        quantity=Decimal("100"),
        stop_price=Decimal("100.20"),
        received_at="2026-09-01T10:30:00.000",
    )
    stop_lim = Order(
        order_id="SL1",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        quantity=Decimal("100"),
        stop_price=Decimal("100.20"),
        limit_price=Decimal("100.10"),
        received_at="2026-09-01T10:30:00.000",
    )

    assert classify_order_category(mkt, quote) == OrderTypeCategory.MARKET
    assert classify_order_category(stop, quote) == OrderTypeCategory.STOP
    assert classify_order_category(stop_lim, quote) == OrderTypeCategory.STOP


def test_classify_order_category_buy_limit_boundaries():
    from finrl.domain.quote import Quote
    from finrl.rules.classification import OrderTypeCategory, classify_order_category

    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.000",
    )
    # NBBO Midpoint = 100.00

    buy_marketable = make_order(OrderSide.BUY, OrderType.LIMIT, Decimal("100.10"))
    buy_midpoint_better = make_order(OrderSide.BUY, OrderType.LIMIT, Decimal("100.00"))
    buy_midpoint_inside = make_order(OrderSide.BUY, OrderType.LIMIT, Decimal("100.05"))
    buy_non_marketable = make_order(OrderSide.BUY, OrderType.LIMIT, Decimal("99.95"))

    assert classify_order_category(buy_marketable, quote) == OrderTypeCategory.MARKETABLE_LIMIT
    assert classify_order_category(buy_midpoint_better, quote) == OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT
    assert classify_order_category(buy_midpoint_inside, quote) == OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT
    assert classify_order_category(buy_non_marketable, quote) == OrderTypeCategory.NON_MARKETABLE_LIMIT


def test_classify_order_category_sell_limit_boundaries():
    from finrl.domain.quote import Quote
    from finrl.rules.classification import OrderTypeCategory, classify_order_category

    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.000",
    )
    # NBBO Midpoint = 100.00

    sell_marketable = make_order(OrderSide.SELL, OrderType.LIMIT, Decimal("99.90"))
    sell_midpoint_better = make_order(OrderSide.SELL, OrderType.LIMIT, Decimal("100.00"))
    sell_midpoint_inside = make_order(OrderSide.SELL, OrderType.LIMIT, Decimal("99.95"))
    sell_non_marketable = make_order(OrderSide.SELL, OrderType.LIMIT, Decimal("100.05"))

    assert classify_order_category(sell_marketable, quote) == OrderTypeCategory.MARKETABLE_LIMIT
    assert classify_order_category(sell_midpoint_better, quote) == OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT
    assert classify_order_category(sell_midpoint_inside, quote) == OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT
    assert classify_order_category(sell_non_marketable, quote) == OrderTypeCategory.NON_MARKETABLE_LIMIT


def test_classify_order_category_missing_quote():
    from finrl.rules.classification import classify_order_category
    import pytest

    order = make_order(OrderSide.BUY, OrderType.LIMIT, Decimal("100.00"))
    with pytest.raises(ValueError, match="Receipt quote is required"):
        classify_order_category(order, None)


