from decimal import Decimal

from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote
from finrl.rules.metrics import (
    effective_spread,
    midpoint,
    price_improvement,
    quoted_spread,
)


def make_quote() -> Quote:
    return Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.000",
    )


def test_midpoint():
    quote = make_quote()

    assert midpoint(quote) == Decimal("100.00")


def test_quoted_spread():
    quote = make_quote()

    assert quoted_spread(quote) == Decimal("0.20")


def test_buy_price_improvement():
    quote = make_quote()

    improvement = price_improvement(
        OrderSide.BUY,
        Decimal("100.00"),
        quote,
    )

    assert improvement == Decimal("0.10")


def test_sell_price_improvement():
    quote = make_quote()

    improvement = price_improvement(
        OrderSide.SELL,
        Decimal("100.00"),
        quote,
    )

    assert improvement == Decimal("0.10")


def test_effective_spread():
    quote = make_quote()

    spread = effective_spread(
        OrderSide.BUY,
        Decimal("100.05"),
        quote,
    )

    assert spread == Decimal("0.10")
