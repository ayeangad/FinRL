from datetime import datetime, timedelta
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote
from finrl.rules.execution_context import (
    quote_at_horizon,
    quote_for_execution,
    realized_spread_at_horizon,
)


def test_quote_for_execution_uses_execution_timestamp():
    execution = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("100"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.200"
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
                    "2026-09-01T10:30:00.100"
                ),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.80"),
                bid_size=Decimal("400"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("200"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.150"
                ),
            ),
        ],
    )

    quote = quote_for_execution(execution, market)

    assert quote is not None
    assert quote.bid_price == Decimal("99.80")
    assert quote.ask_price == Decimal("100.20")


def test_quote_for_latest_execution_does_not_depend_on_execution_order():
    executions = [
        Execution(
            execution_id="EXE-0002",
            order_id="ORD-0001",
            price=Decimal("100.10"),
            quantity=Decimal("20"),
            executed_at=datetime.fromisoformat(
                "2026-09-01T10:30:00.300"
            ),
        ),
        Execution(
            execution_id="EXE-0001",
            order_id="ORD-0001",
            price=Decimal("100.00"),
            quantity=Decimal("40"),
            executed_at=datetime.fromisoformat(
                "2026-09-01T10:30:00.200"
            ),
        ),
    ]

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
                    "2026-09-01T10:30:00.150"
                ),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.80"),
                bid_size=Decimal("400"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("200"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.250"
                ),
            ),
        ],
    )

    latest_execution = max(
        executions,
        key=lambda execution: execution.executed_at,
    )

    quote = quote_for_execution(
        latest_execution,
        market,
    )

    assert quote is not None
    assert quote.bid_price == Decimal("99.80")
    assert quote.ask_price == Decimal("100.20")


def test_quote_at_horizon_returns_latest_quote_at_horizon():
    execution = Execution(
        execution_id="EXE-0003",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("100"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.200"
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
                    "2026-09-01T10:30:00.250"
                ),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.80"),
                bid_size=Decimal("400"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("200"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.300"
                ),
            ),
        ],
    )

    quote = quote_at_horizon(
        execution,
        market,
        timedelta(milliseconds=100),
    )

    assert quote is not None
    assert quote.bid_price == Decimal("99.80")
    assert quote.ask_price == Decimal("100.20")


def test_quote_at_horizon_accepts_exact_horizon_timestamp():
    execution = Execution(
        execution_id="EXE-0004",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("100"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.200"
        ),
    )

    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.85"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.15"),
        ask_size=Decimal("300"),
        timestamp=datetime.fromisoformat(
            "2026-09-01T10:30:00.300"
        ),
    )

    market = MarketState(
        security="FINRL",
        quotes=[quote],
    )

    result = quote_at_horizon(
        execution,
        market,
        timedelta(milliseconds=100),
    )

    assert result == quote


def test_quote_at_horizon_returns_none_when_no_quote_exists():
    execution = Execution(
        execution_id="EXE-0005",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("100"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.200"
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
                    "2026-09-01T10:30:00.350"
                ),
            )
        ],
    )

    result = quote_at_horizon(
        execution,
        market,
        timedelta(milliseconds=100),
    )

    assert result is None


def test_realized_spread_at_horizon_multiple_executions_different_quotes():
    # EX1: executed at 10:30:00.200, price 100.00, qty 40. Horizon +100ms -> target 10:30:00.300
    # Quote at 10:30:00.300: 100.10 / 100.30 (midpoint 100.20)
    # RS1 = 2 * (100.00 - 100.20) = -0.40
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("40"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.200"
        ),
    )

    # EX2: executed at 10:30:00.300, price 100.30, qty 20. Horizon +100ms -> target 10:30:00.400
    # Quote at 10:30:00.400: 100.00 / 100.20 (midpoint 100.10)
    # RS2 = 2 * (100.30 - 100.10) = +0.40
    ex2 = Execution(
        execution_id="EXE-0002",
        order_id="ORD-0001",
        price=Decimal("100.30"),
        quantity=Decimal("20"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.300"
        ),
    )

    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("100.10"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.30"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.300"
                ),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("100.00"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.400"
                ),
            ),
        ],
    )

    result = realized_spread_at_horizon(
        OrderSide.BUY,
        [ex1, ex2],
        market,
        timedelta(milliseconds=100),
    )

    assert result == (
        Decimal("40") * Decimal("-0.40")
        + Decimal("20") * Decimal("0.40")
    ) / Decimal("60")


def test_realized_spread_at_horizon_missing_future_quote():
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("40"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.200"
        ),
    )
    ex2 = Execution(
        execution_id="EXE-0002",
        order_id="ORD-0001",
        price=Decimal("100.30"),
        quantity=Decimal("20"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.300"
        ),
    )

    # Market only has a quote valid for EX1 (at 10:30:00.300), but EX2 target is 10:30:00.400,
    # and market quote is after EX2 target or missing.
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("100.10"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.30"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.100"  # Before EX1 & EX2 execution
                ),
            ),
        ],
    )

    # No quote exists for horizon +100ms when quote is before execution
    # EX1 target: 10:30:00.300 -> quote at 10:30:00.100 valid
    # But if no quote at all for EX2 target, let's test quote after target
    market_missing = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("100.10"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.30"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.250"  # Valid for EX1 target (0.300), invalid for EX2 if we require quote strictly after EX2? Wait, quote at 0.250 IS valid for 0.400 too!
                ),
            ),
        ],
    )
    # If quote is at 10:30:00.500:
    # EX1 target 10:30:00.300 -> quote at 10:30:00.500 is > 0.300 -> None!
    market_future_only = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("100.10"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.30"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.500"
                ),
            ),
        ],
    )

    result = realized_spread_at_horizon(
        OrderSide.BUY,
        [ex1, ex2],
        market_future_only,
        timedelta(milliseconds=100),
    )

    assert result is None


def test_realized_spread_at_horizon_exact_boundary():
    ex = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("100"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.200"
        ),
    )

    # Exact horizon target: 10:30:00.300
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("100.10"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.30"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.300"
                ),
            ),
        ],
    )

    result = realized_spread_at_horizon(
        OrderSide.BUY,
        [ex],
        market,
        timedelta(milliseconds=100),
    )

    # RS = 2 * (100.00 - 100.20) = -0.40
    assert result == Decimal("-0.40")


