from datetime import datetime
from decimal import Decimal

from finrl.domain.market import MarketState
from finrl.domain.quote import Quote


def test_quote_at_returns_latest_valid_quote():
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.99"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.01"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.000"
                ),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.98"),
                bid_size=Decimal("400"),
                ask_price=Decimal("100.02"),
                ask_size=Decimal("200"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.100"
                ),
            ),
        ],
    )

    quote = market.quote_at(
        datetime.fromisoformat("2026-09-01T10:30:00.150")
    )

    assert quote is not None
    assert quote.bid_price == Decimal("99.98")
    assert quote.ask_price == Decimal("100.02")


def test_quote_at_returns_none_before_first_quote():
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.99"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.01"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.000"
                ),
            )
        ],
    )

    quote = market.quote_at(
        datetime.fromisoformat("2026-09-01T10:29:59.999")
    )

    assert quote is None
