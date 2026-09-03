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


def test_buy_non_marketable_limit_becomes_executable_when_bid_reaches_limit():
    # Receipt quote: 99.80 / 99.90 -> BUY limit 100.00 is marketable at receipt!
    # Let's test non-marketable at receipt: BUY limit 99.85 when receipt is 99.70 / 99.90.
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")
    order = Order(
        order_id="ORD-LIMIT-BUY",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("99.85"),
        received_at=received_at,
    )

    market = MarketState(
        security="FINRL",
        quotes=[
            # Receipt quote: 99.70 bid / 99.90 ask -> non-marketable at receipt (limit 99.85 < ask 99.90)
            Quote(
                security="FINRL",
                bid_price=Decimal("99.70"),
                bid_size=Decimal("500"),
                ask_price=Decimal("99.90"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            # T+100ms: 99.80 bid / 99.95 ask -> still bid 99.80 < limit 99.85
            Quote(
                security="FINRL",
                bid_price=Decimal("99.80"),
                bid_size=Decimal("500"),
                ask_price=Decimal("99.95"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
            # T+200ms: 99.85 bid / 100.00 ask -> bid 99.85 >= limit 99.85 -> EXECUTABLE!
            Quote(
                security="FINRL",
                bid_price=Decimal("99.85"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.00"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.300"),
            ),
        ],
    )

    assert executable_time(order, market) == datetime.fromisoformat("2026-09-01T10:30:00.300")


def test_sell_non_marketable_limit_becomes_executable_when_ask_reaches_limit():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")
    order = Order(
        order_id="ORD-LIMIT-SELL",
        security="FINRL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.10"),
        received_at=received_at,
    )

    market = MarketState(
        security="FINRL",
        quotes=[
            # Receipt quote: 99.90 bid / 100.20 ask -> non-marketable at receipt (limit 100.10 > bid 99.90)
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            # T+100ms: 99.95 bid / 100.15 ask -> ask 100.15 > limit 100.10
            Quote(
                security="FINRL",
                bid_price=Decimal("99.95"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.15"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
            # T+200ms: 100.00 bid / 100.10 ask -> ask 100.10 <= limit 100.10 -> EXECUTABLE!
            Quote(
                security="FINRL",
                bid_price=Decimal("100.00"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.300"),
            ),
        ],
    )

    assert executable_time(order, market) == datetime.fromisoformat("2026-09-01T10:30:00.300")


def test_stop_limit_requires_both_trigger_and_limit_executability():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")
    order = Order(
        order_id="ORD-STOP-LIMIT",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        quantity=Decimal("100"),
        stop_price=Decimal("100.00"),
        limit_price=Decimal("99.90"),  # non-marketable limit when triggered at 100.00 ask
        received_at=received_at,
    )

    market = MarketState(
        security="FINRL",
        quotes=[
            # Receipt quote: 99.70 bid / 99.90 ask (not triggered: ask 99.90 < stop 100.00)
            Quote(
                security="FINRL",
                bid_price=Decimal("99.70"),
                bid_size=Decimal("500"),
                ask_price=Decimal("99.90"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            # T+100ms: 99.75 bid / 100.05 ask -> TRIGGERED (ask 100.05 >= stop 100.00).
            # But limit check: bid 99.75 < limit 99.90 and ask 100.05 > limit 99.90 -> limit NOT executable yet.
            Quote(
                security="FINRL",
                bid_price=Decimal("99.75"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.05"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
            # T+200ms: 99.90 bid / 100.10 ask -> bid 99.90 >= limit 99.90 -> EXECUTABLE!
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.300"),
            ),
        ],
    )

    assert executable_time(order, market) == datetime.fromisoformat("2026-09-01T10:30:00.300")