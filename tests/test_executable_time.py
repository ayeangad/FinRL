from datetime import datetime
from decimal import Decimal
import pytest

from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote
from finrl.rules.executable_time import executable_time


RECEIVED_AT = datetime.fromisoformat(
    "2026-09-01T10:30:00.100"
)


def make_order(
    order_type: OrderType,
    side: OrderSide = OrderSide.BUY,
    stop_price: Decimal | None = None,
) -> Order:
    return Order(
        order_id="ORD-0001",
        security="FINRL",
        side=side,
        order_type=order_type,
        quantity=Decimal("100"),
        limit_price=(
            Decimal("100.00")
            if order_type == OrderType.LIMIT
            else None
        ),
        stop_price=stop_price,
        received_at=RECEIVED_AT,
    )


def make_market() -> MarketState:
    return MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.80"),
                bid_size=Decimal("500"),
                ask_price=Decimal("99.90"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.200"
                ),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("400"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("200"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.300"
                ),
            ),
        ],
    )


def test_market_order_executable_at_receipt():
    order = make_order(OrderType.MARKET)

    assert executable_time(order) == RECEIVED_AT


def test_limit_order_executable_at_receipt():
    order = make_order(OrderType.LIMIT)

    assert executable_time(order) == RECEIVED_AT


def test_buy_stop_becomes_executable_when_ask_reaches_stop():
    order = make_order(
        OrderType.STOP,
        side=OrderSide.BUY,
        stop_price=Decimal("100.00"),
    )

    market = make_market()

    assert executable_time(order, market) == datetime.fromisoformat(
        "2026-09-01T10:30:00.300"
    )


def test_sell_stop_becomes_executable_when_bid_reaches_stop():
    order = make_order(
        OrderType.STOP,
        side=OrderSide.SELL,
        stop_price=Decimal("99.90"),
    )

    market = make_market()

    assert executable_time(order, market) == datetime.fromisoformat(
        "2026-09-01T10:30:00.200"
    )


def test_stop_order_ignores_quotes_before_order_received():
    order = Order(
        order_id="ORD-0002",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        quantity=Decimal("100"),
        stop_price=Decimal("100.00"),
        received_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.250"
        ),
    )

    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.200"
                ),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("100.00"),
                bid_size=Decimal("400"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("200"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.300"
                ),
            ),
        ],
    )

    assert executable_time(order, market) == datetime.fromisoformat(
        "2026-09-01T10:30:00.300"
    )

def test_stop_order_requires_market():
    order = make_order(
        OrderType.STOP,
        stop_price=Decimal("100.00"),
    )

    with pytest.raises(
        ValueError,
        match="Market state is required",
    ):
        executable_time(order)


def test_stop_order_requires_stop_price():
    order = make_order(OrderType.STOP)

    market = make_market()

    with pytest.raises(
        ValueError,
        match="requires stop_price",
    ):
        executable_time(order, market)


def test_untriggered_stop_order_raises_error():
    order = make_order(
        OrderType.STOP,
        stop_price=Decimal("200.00"),
    )

    market = make_market()

    with pytest.raises(
        ValueError,
        match="was not triggered",
    ):
        executable_time(order, market)