from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote
from finrl.rules.metrics import (
    effective_spread,
    midpoint,
    price_improvement,
    quoted_spread,
    realized_spread,
    share_weighted_effective_spread,
    share_weighted_price_improvement,
    share_weighted_realized_spread,
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


def test_share_weighted_price_improvement_for_partial_executions():
    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.100",
    )

    executions = [
        Execution(
            execution_id="EXE-0001",
            order_id="ORD-0001",
            price=Decimal("100.00"),
            quantity=Decimal("40"),
            executed_at="2026-09-01T10:30:00.200",
        ),
        Execution(
            execution_id="EXE-0002",
            order_id="ORD-0001",
            price=Decimal("100.20"),
            quantity=Decimal("20"),
            executed_at="2026-09-01T10:30:00.300",
        ),
    ]

    result = share_weighted_price_improvement(
        OrderSide.BUY,
        executions,
        quote,
    )

    assert result == (
        Decimal("40") * Decimal("0.10")
        + Decimal("20") * Decimal("-0.10")
    ) / Decimal("60")


def test_share_weighted_effective_spread_for_partial_executions():
    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.100",
    )

    executions = [
        Execution(
            execution_id="EXE-0001",
            order_id="ORD-0001",
            price=Decimal("100.00"),
            quantity=Decimal("40"),
            executed_at="2026-09-01T10:30:00.200",
        ),
        Execution(
            execution_id="EXE-0002",
            order_id="ORD-0001",
            price=Decimal("100.20"),
            quantity=Decimal("20"),
            executed_at="2026-09-01T10:30:00.300",
        ),
    ]

    result = share_weighted_effective_spread(
        OrderSide.BUY,
        executions,
        quote,
    )

    assert result == (
        Decimal("40") * Decimal("0.00")
        + Decimal("20") * Decimal("0.40")
    ) / Decimal("60")


def test_buy_realized_spread_negative():
    future_quote = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.250",
    )

    result = realized_spread(
        OrderSide.BUY,
        Decimal("100.00"),
        future_quote,
    )

    assert result == Decimal("-0.40")


def test_buy_realized_spread_positive():
    future_quote = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.250",
    )

    result = realized_spread(
        OrderSide.BUY,
        Decimal("100.30"),
        future_quote,
    )

    assert result == Decimal("0.20")


def test_sell_realized_spread_zero():
    future_quote = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.250",
    )

    result = realized_spread(
        OrderSide.SELL,
        Decimal("100.20"),
        future_quote,
    )

    assert result == Decimal("0.00")


def test_share_weighted_realized_spread_buy_order():
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("40"),
        executed_at="2026-09-01T10:30:00.200",
    )
    ex2 = Execution(
        execution_id="EXE-0002",
        order_id="ORD-0001",
        price=Decimal("100.30"),
        quantity=Decimal("20"),
        executed_at="2026-09-01T10:30:00.300",
    )

    # RS1 = 2 * (100.00 - 100.20) = -0.40
    q1 = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.250",
    )
    # RS2 = 2 * (100.30 - 100.10) = +0.40
    q2 = Quote(
        security="FINRL",
        bid_price=Decimal("100.00"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.20"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.350",
    )

    future_quotes = {
        "EXE-0001": q1,
        "EXE-0002": q2,
    }

    result = share_weighted_realized_spread(
        OrderSide.BUY,
        [ex1, ex2],
        future_quotes,
    )

    assert result == (
        Decimal("40") * Decimal("-0.40")
        + Decimal("20") * Decimal("0.40")
    ) / Decimal("60")


def test_share_weighted_realized_spread_sell_order():
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.20"),
        quantity=Decimal("40"),
        executed_at="2026-09-01T10:30:00.200",
    )
    ex2 = Execution(
        execution_id="EXE-0002",
        order_id="ORD-0001",
        price=Decimal("99.80"),
        quantity=Decimal("20"),
        executed_at="2026-09-01T10:30:00.300",
    )

    # Future midpoint = 100.00
    q = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.350",
    )

    # RS1 = 2 * (100.00 - 100.20) = -0.40
    # RS2 = 2 * (100.00 - 99.80) = +0.40
    future_quotes = {
        "EXE-0001": q,
        "EXE-0002": q,
    }

    result = share_weighted_realized_spread(
        OrderSide.SELL,
        [ex1, ex2],
        future_quotes,
    )

    assert result == (
        Decimal("40") * Decimal("-0.40")
        + Decimal("20") * Decimal("0.40")
    ) / Decimal("60")


def test_share_weighted_realized_spread_missing_execution_id():
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("40"),
        executed_at="2026-09-01T10:30:00.200",
    )
    ex2 = Execution(
        execution_id="EXE-0002",
        order_id="ORD-0001",
        price=Decimal("100.30"),
        quantity=Decimal("20"),
        executed_at="2026-09-01T10:30:00.300",
    )

    q1 = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.250",
    )

    # Missing EXE-0002
    future_quotes = {"EXE-0001": q1}

    result = share_weighted_realized_spread(
        OrderSide.BUY,
        [ex1, ex2],
        future_quotes,
    )

    assert result is None


def test_share_weighted_realized_spread_explicit_none_quote():
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("40"),
        executed_at="2026-09-01T10:30:00.200",
    )

    q1 = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.250",
    )

    future_quotes = {
        "EXE-0001": q1,
        "EXE-0002": None,
    }

    result = share_weighted_realized_spread(
        OrderSide.BUY,
        [ex1],
        future_quotes,
    )

    assert result == Decimal("-0.40")


def test_share_weighted_realized_spread_empty_executions():
    assert share_weighted_realized_spread(
        OrderSide.BUY,
        [],
        {},
    ) is None


def test_percentage_effective_spread():
    from finrl.rules.metrics import percentage_effective_spread

    receipt_quote = make_quote()  # 99.90 / 100.10 -> midpoint = 100.00

    # Effective spread $ = 2 * |100.05 - 100.00| = 0.10
    # Effective spread % = 0.10 / 100.00 = 0.001
    assert percentage_effective_spread(OrderSide.BUY, Decimal("100.05"), receipt_quote) == Decimal("0.001")

    # Zero effective spread $ = 2 * |100.00 - 100.00| = 0.00 -> % = 0.000
    assert percentage_effective_spread(OrderSide.BUY, Decimal("100.00"), receipt_quote) == Decimal("0")


def test_percentage_quoted_spread():
    from finrl.rules.metrics import percentage_quoted_spread

    receipt_quote = make_quote()  # 99.90 / 100.10 -> quoted $ = 0.20, midpoint = 100.00
    # Quoted spread % = 0.20 / 100.00 = 0.002
    assert percentage_quoted_spread(receipt_quote) == Decimal("0.002")


def test_percentage_realized_spread_uses_receipt_midpoint_denominator():
    from finrl.rules.metrics import percentage_realized_spread

    receipt_quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.000",
    )  # receipt midpoint = 100.00

    future_quote = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:01.000",
    )  # future midpoint = 100.20

    # BUY execution = 100.00
    # realized $ = 2 * (100.00 - 100.20) = -0.40
    # realized % = -0.40 / 100.00 = -0.004 (verifying receipt midpoint 100.00 is denominator, NOT 100.20!)
    result_neg = percentage_realized_spread(
        OrderSide.BUY,
        Decimal("100.00"),
        future_quote,
        receipt_quote,
    )
    assert result_neg == Decimal("-0.004")

    # BUY execution = 100.30
    # realized $ = 2 * (100.30 - 100.20) = +0.20
    # realized % = +0.20 / 100.00 = +0.002
    result_pos = percentage_realized_spread(
        OrderSide.BUY,
        Decimal("100.30"),
        future_quote,
        receipt_quote,
    )
    assert result_pos == Decimal("0.002")


def test_share_weighted_percentage_effective_spread():
    from finrl.rules.metrics import share_weighted_percentage_effective_spread

    receipt_quote = make_quote()  # midpoint = 100.00
    ex1 = Execution(execution_id="E1", order_id="O1", price=Decimal("100.00"), quantity=Decimal("40"), executed_at="2026-09-01T10:30:00.100")  # %ES = 0
    ex2 = Execution(execution_id="E2", order_id="O1", price=Decimal("100.20"), quantity=Decimal("20"), executed_at="2026-09-01T10:30:00.200")  # %ES = 0.40/100 = 0.004

    # (40 * 0 + 20 * 0.004) / 60 = 0.08 / 60 = 0.0013333...
    res = share_weighted_percentage_effective_spread(OrderSide.BUY, [ex1, ex2], receipt_quote)
    assert res == (Decimal("40") * Decimal("0") + Decimal("20") * Decimal("0.004")) / Decimal("60")


def test_share_weighted_percentage_realized_spread():
    from finrl.rules.metrics import share_weighted_percentage_realized_spread

    receipt_quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp="2026-09-01T10:30:00.000",
    )  # receipt midpoint = 100.00

    ex1 = Execution(execution_id="E1", order_id="O1", price=Decimal("100.00"), quantity=Decimal("40"), executed_at="2026-09-01T10:30:00.100")
    ex2 = Execution(execution_id="E2", order_id="O1", price=Decimal("100.30"), quantity=Decimal("20"), executed_at="2026-09-01T10:30:00.200")

    future_q1 = Quote(security="FINRL", bid_price=Decimal("100.10"), bid_size=Decimal("500"), ask_price=Decimal("100.30"), ask_size=Decimal("300"), timestamp="2026-09-01T10:30:01.000")
    future_q2 = Quote(security="FINRL", bid_price=Decimal("100.00"), bid_size=Decimal("500"), ask_price=Decimal("100.20"), ask_size=Decimal("300"), timestamp="2026-09-01T10:30:01.000")

    # %RS1 = -0.40 / 100.00 = -0.004
    # %RS2 = +0.40 / 100.00 = +0.004
    future_quotes = {"E1": future_q1, "E2": future_q2}

    res = share_weighted_percentage_realized_spread(OrderSide.BUY, [ex1, ex2], future_quotes, receipt_quote)
    assert res == (Decimal("40") * Decimal("-0.004") + Decimal("20") * Decimal("0.004")) / Decimal("60")





