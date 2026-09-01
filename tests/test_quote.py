from datetime import datetime
from decimal import Decimal

from finrl.domain.quote import Quote


def test_create_quote():
    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.99"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.01"),
        ask_size=Decimal("300"),
        timestamp=datetime.fromisoformat(
            "2026-09-01T10:30:00.100"
        ),
    )

    assert quote.security == "FINRL"
    assert quote.bid_price == Decimal("99.99")
    assert quote.ask_price == Decimal("100.01")
    assert quote.bid_size == Decimal("500")
    assert quote.ask_size == Decimal("300")
