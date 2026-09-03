from datetime import datetime
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote
from finrl.evals.order_evaluator import evaluate_order
from finrl.rules.horizons import RealizedSpreadHorizon


def test_realized_spread_end_to_end_integration():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.000")

    order = Order(
        order_id="ORD-INT-001",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.50"),
        received_at=received_at,
    )

    # Partial execution 1: 60 shares @ 100.10 executed at 10:30:00.100
    # Partial execution 2: 40 shares @ 100.40 executed at 10:30:00.200
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-INT-001",
        price=Decimal("100.10"),
        quantity=Decimal("60"),
        executed_at=datetime.fromisoformat("2026-09-01T10:30:00.100"),
    )
    ex2 = Execution(
        execution_id="EXE-0002",
        order_id="ORD-INT-001",
        price=Decimal("100.40"),
        quantity=Decimal("40"),
        executed_at=datetime.fromisoformat("2026-09-01T10:30:00.200"),
    )

    market = MarketState(
        security="FINRL",
        quotes=[
            # Receipt quote: 10:30:00.000 -> 99.90 / 100.10 (mid 100.00)
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            # Quote for EX1 50ms horizon (target 10:30:00.150): 100.00 / 100.20 (mid 100.10)
            # EX1 RS_50ms = 2 * (100.10 - 100.10) = 0.00
            Quote(
                security="FINRL",
                bid_price=Decimal("100.00"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.150"),
            ),
            # Quote for EX2 50ms horizon (target 10:30:00.250): 100.20 / 100.40 (mid 100.30)
            # EX2 RS_50ms = 2 * (100.40 - 100.30) = +0.20
            Quote(
                security="FINRL",
                bid_price=Decimal("100.20"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.40"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.250"),
            ),
            # Quote for 1s horizon (EX1 target 10:30:01.100, EX2 target 10:30:01.200):
            # Midpoint = 100.00
            # EX1 RS_1s = 2 * (100.10 - 100.00) = +0.20
            # EX2 RS_1s = 2 * (100.40 - 100.00) = +0.80
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:01.000"),
            ),
            # Quote up to 15s horizon (target 10:30:15.100 / 10:30:15.200):
            # Midpoint = 99.80
            # EX1 RS_15s = 2 * (100.10 - 99.80) = +0.60
            # EX2 RS_15s = 2 * (100.40 - 99.80) = +1.20
            Quote(
                security="FINRL",
                bid_price=Decimal("99.70"),
                bid_size=Decimal("500"),
                ask_price=Decimal("99.90"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:10.000"),
            ),
            # Market quotes end here. No quotes available for 1m (10:31:00+) or 5m (10:35:00+).
        ],
    )

    report = evaluate_order(
        order,
        [ex1, ex2],
        market,
    )

    # 50ms: (60 * 0.00 + 40 * 0.20) / 100 = 0.08
    assert report.realized_spreads[RealizedSpreadHorizon.MS_50] == Decimal("0.08")

    # 1s: (60 * 0.20 + 40 * 0.80) / 100 = (12 + 32) / 100 = 0.44
    assert report.realized_spreads[RealizedSpreadHorizon.S_1] == Decimal("0.44")

    # 15s, 1m & 5m target timestamps all occur after 10:30:10.000, so they look up the prevailing NBBO at 10:30:10.000:
    assert report.realized_spreads[RealizedSpreadHorizon.S_15] == Decimal("0.84")
    assert report.realized_spreads[RealizedSpreadHorizon.M_1] == Decimal("0.84")
    assert report.realized_spreads[RealizedSpreadHorizon.M_5] == Decimal("0.84")


def test_realized_spread_integration_missing_quote_propagates_none():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.000")

    order = Order(
        order_id="ORD-INT-002",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.50"),
        received_at=received_at,
    )

    # Execution at 10:30:00.100 (target 50ms = 10:30:00.150)
    ex = Execution(
        execution_id="EXE-0001",
        order_id="ORD-INT-002",
        price=Decimal("100.10"),
        quantity=Decimal("100"),
        executed_at=datetime.fromisoformat("2026-09-01T10:30:00.100"),
    )

    # Market stream only has a quote at receipt (10:30:00.000) and a quote in the far future (10:30:05.000).
    # Wait, quote at 10:30:00.000 is <= 10:30:00.150!
    # If market stream only starts at 10:30:00.500:
    market_late_start = MarketState(
        security="FINRL",
        quotes=[
            # First quote is at 10:30:00.500
            Quote(
                security="FINRL",
                bid_price=Decimal("100.00"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.500"),
            ),
        ],
    )

    report = evaluate_order(
        order,
        [ex],
        market_late_start,
    )

    # No receipt quote (order received at 0.000, quote starts at 0.500) -> unevaluated report with None realized spreads
    assert report.price_improvement is None
    assert report.realized_spreads[RealizedSpreadHorizon.MS_50] is None
    assert report.realized_spreads[RealizedSpreadHorizon.S_1] is None

