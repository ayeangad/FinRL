from decimal import Decimal

from finrl.rules.classification import OrderTypeCategory
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport
from finrl.rules.report_statistics import (
    share_weighted_effective_spread,
    share_weighted_percentage_effective_spread,
    share_weighted_percentage_quoted_spread,
    share_weighted_percentage_realized_spread,
    share_weighted_price_improvement,
    share_weighted_quoted_spread,
    share_weighted_realized_spread,
)


def make_report(
    order_id: str,
    requested_qty: str = "100",
    executed_qty: str = "100",
    price_improvement: str | None = None,
    effective_spread: str | None = None,
    quoted_spread: str | None = None,
    realized_spreads: dict[RealizedSpreadHorizon, Decimal | None] | None = None,
    percentage_effective_spread: str | None = None,
    percentage_quoted_spread: str | None = None,
    percentage_realized_spreads: dict[RealizedSpreadHorizon, Decimal | None] | None = None,
) -> OrderReport:
    return OrderReport(
        order_id=order_id,
        order_size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        order_type_category=OrderTypeCategory.MARKET,
        reportable=True,
        requested_quantity=Decimal(requested_qty),
        executed_quantity=Decimal(executed_qty),
        price_improvement=(
            Decimal(price_improvement) if price_improvement is not None else None
        ),
        effective_spread=(
            Decimal(effective_spread) if effective_spread is not None else None
        ),
        quoted_spread=(
            Decimal(quoted_spread) if quoted_spread is not None else None
        ),
        realized_spreads=realized_spreads or {},
        percentage_effective_spread=(
            Decimal(percentage_effective_spread)
            if percentage_effective_spread is not None
            else None
        ),
        percentage_quoted_spread=(
            Decimal(percentage_quoted_spread)
            if percentage_quoted_spread is not None
            else None
        ),
        percentage_realized_spreads=percentage_realized_spreads or {},
    )


def test_normal_share_weighted_aggregation():
    r1 = make_report("R1", executed_qty="100", price_improvement="0.10", effective_spread="0.10", quoted_spread="0.20")
    r2 = make_report("R2", executed_qty="300", price_improvement="0.20", effective_spread="0.20", quoted_spread="0.40")

    # PI: (100*0.10 + 300*0.20)/400 = (10 + 60)/400 = 70/400 = 0.175
    assert share_weighted_price_improvement([r1, r2]) == Decimal("0.175")
    # ES: 0.175
    assert share_weighted_effective_spread([r1, r2]) == Decimal("0.175")
    # QS: (100*0.20 + 300*0.40)/400 = (20 + 120)/400 = 140/400 = 0.35
    assert share_weighted_quoted_spread([r1, r2]) == Decimal("0.35")


def test_aggregation_ignores_none_metrics():
    r1 = make_report("R1", executed_qty="100", effective_spread="0.10")
    r2 = make_report("R2", executed_qty="200", effective_spread=None)
    r3 = make_report("R3", executed_qty="100", effective_spread="0.20")

    # (100*0.10 + 100*0.20)/(100+100) = 30/200 = 0.15
    assert share_weighted_effective_spread([r1, r2, r3]) == Decimal("0.15")


def test_aggregation_zero_executed_quantity_returns_none():
    r1 = make_report("R1", executed_qty="0", effective_spread="0.10")
    r2 = make_report("R2", executed_qty="0", effective_spread=None)

    assert share_weighted_effective_spread([r1, r2]) is None


def test_all_metrics_none_returns_none():
    r1 = make_report("R1", executed_qty="100", effective_spread=None)
    r2 = make_report("R2", executed_qty="200", effective_spread=None)

    assert share_weighted_effective_spread([r1, r2]) is None


def test_empty_population_returns_none():
    assert share_weighted_price_improvement([]) is None
    assert share_weighted_effective_spread([]) is None
    assert share_weighted_quoted_spread([]) is None
    assert share_weighted_realized_spread([], RealizedSpreadHorizon.MS_50) is None
    assert share_weighted_percentage_effective_spread([]) is None
    assert share_weighted_percentage_quoted_spread([]) is None
    assert share_weighted_percentage_realized_spread([], RealizedSpreadHorizon.MS_50) is None


def test_realized_spread_multiple_horizons():
    r1 = make_report(
        "R1",
        executed_qty="100",
        realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.05"),
            RealizedSpreadHorizon.S_1: Decimal("0.10"),
        },
    )
    r2 = make_report(
        "R2",
        executed_qty="300",
        realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.15"),
            RealizedSpreadHorizon.S_1: None,
        },
    )

    # 50ms RS: (100*0.05 + 300*0.15)/400 = (5 + 45)/400 = 50/400 = 0.125
    assert share_weighted_realized_spread([r1, r2], RealizedSpreadHorizon.MS_50) == Decimal("0.125")

    # 1s RS: (100*0.10)/100 = 0.10 (since r2 is None)
    assert share_weighted_realized_spread([r1, r2], RealizedSpreadHorizon.S_1) == Decimal("0.10")

    # 15s RS: both empty -> None
    assert share_weighted_realized_spread([r1, r2], RealizedSpreadHorizon.S_15) is None


def test_percentage_spread_aggregation_different_midpoints():
    # R1: stock at $10.00, ES=$0.10 -> %ES = 0.0100 (100 shares)
    r1 = make_report(
        "R1",
        executed_qty="100",
        effective_spread="0.10",
        percentage_effective_spread="0.0100",
        percentage_quoted_spread="0.0200",
        percentage_realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.0050"),
            RealizedSpreadHorizon.S_1: None,
        },
    )
    # R2: stock at $100.00, ES=$0.10 -> %ES = 0.0010 (300 shares)
    r2 = make_report(
        "R2",
        executed_qty="300",
        effective_spread="0.10",
        percentage_effective_spread="0.0010",
        percentage_quoted_spread="0.0040",
        percentage_realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.0010"),
            RealizedSpreadHorizon.S_1: Decimal("0.0020"),
        },
    )

    # Aggregated %ES = (100 * 0.0100 + 300 * 0.0010) / 400 = (1.0 + 0.3) / 400 = 1.3 / 400 = 0.00325
    assert share_weighted_percentage_effective_spread([r1, r2]) == Decimal("0.00325")

    # Aggregated %QS = (100 * 0.0200 + 300 * 0.0040) / 400 = (2.0 + 1.2) / 400 = 3.2 / 400 = 0.008
    assert share_weighted_percentage_quoted_spread([r1, r2]) == Decimal("0.008")

    # Aggregated %RS 50ms = (100 * 0.0050 + 300 * 0.0010) / 400 = (0.5 + 0.3) / 400 = 0.8 / 400 = 0.002
    assert share_weighted_percentage_realized_spread([r1, r2], RealizedSpreadHorizon.MS_50) == Decimal("0.002")

    # Aggregated %RS 1s = (300 * 0.0020) / 300 = 0.0020 (r1 is None)
    assert share_weighted_percentage_realized_spread([r1, r2], RealizedSpreadHorizon.S_1) == Decimal("0.0020")
