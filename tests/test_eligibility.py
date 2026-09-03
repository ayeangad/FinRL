from datetime import datetime
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote
from finrl.rules.eligibility import is_reportable_order


RECEIVED_AT = datetime.fromisoformat("2026-09-01T10:30:00.100")


def make_market_with_quotes() -> MarketState:
    return MarketState(
        security="FINRL",
        quotes=[
            # Receipt quote: 99.90 / 100.10
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=RECEIVED_AT,
            ),
            # Quote at T+100ms: 100.00 / 100.20
            Quote(
                security="FINRL",
                bid_price=Decimal("100.00"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
        ],
    )


def test_market_order_is_reportable():
    order = Order(
        order_id="ORD-0001",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is True


def test_order_without_receipt_quote_is_not_reportable():
    order = Order(
        order_id="ORD-0002",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        received_at=datetime.fromisoformat("2026-09-01T10:29:00.000"),
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is False


def test_marketable_limit_order_is_reportable():
    order = Order(
        order_id="ORD-0003",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.10"),  # Marketable at ask 100.10
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is True


def test_non_marketable_limit_order_becoming_executable_is_reportable():
    # BUY limit 100.00: non-marketable at receipt (ask 100.10), but bid reaches 100.00 at T+100ms
    order = Order(
        order_id="ORD-0004",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.00"),
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is True


def test_non_marketable_limit_order_never_becoming_executable_is_not_reportable():
    # BUY limit 99.50: bid (99.90 -> 100.00) never reaches <= 99.50
    order = Order(
        order_id="ORD-0005",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("99.50"),
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is False


def test_stop_order_becoming_executable_is_reportable():
    # BUY STOP 100.20: ask reaches 100.20 at T+100ms
    order = Order(
        order_id="ORD-0006",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        quantity=Decimal("100"),
        stop_price=Decimal("100.20"),
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is True


def test_untriggered_stop_order_is_not_reportable():
    # BUY STOP 105.00: ask never reaches 105.00
    order = Order(
        order_id="ORD-0007",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        quantity=Decimal("100"),
        stop_price=Decimal("105.00"),
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is False


def test_stop_limit_order_becoming_executable_is_reportable():
    # BUY STOP_LIMIT: stop 100.20 (triggered at T+100ms), limit 100.00 (bid at T+100ms is 100.00 -> executable)
    order = Order(
        order_id="ORD-0008",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        quantity=Decimal("100"),
        stop_price=Decimal("100.20"),
        limit_price=Decimal("100.00"),
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is True


def test_stop_limit_order_triggering_but_limit_never_executable_is_not_reportable():
    # BUY STOP_LIMIT: stop 100.20 (triggered at T+100ms ask 100.20), limit 99.50 (bid 100.00 never reaches <= 99.50)
    order = Order(
        order_id="ORD-0009",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        quantity=Decimal("100"),
        stop_price=Decimal("100.20"),
        limit_price=Decimal("99.50"),
        received_at=RECEIVED_AT,
    )
    market = make_market_with_quotes()

    assert is_reportable_order(order, [], market) is False
