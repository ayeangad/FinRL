from decimal import Decimal

from finrl.rules.category_report import CategoryReport, build_category_report
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport


def make_report(
    order_id: str,
    size_bucket: OrderSizeBucket = OrderSizeBucket.SHARES_100_TO_499,
    reportable: bool = True,
    requested_qty: str = "100",
    executed_qty: str = "100",
    price_improvement: str | None = None,
    effective_spread: str | None = None,
    quoted_spread: str | None = None,
    realized_spreads: dict[RealizedSpreadHorizon, Decimal | None] | None = None,
) -> OrderReport:
    return OrderReport(
        order_id=order_id,
        order_size_bucket=size_bucket,
        reportable=reportable,
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
    )


def test_build_category_report_single_report():
    r1 = make_report(
        "ORD-001",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        requested_qty="200",
        executed_qty="200",
        price_improvement="0.10",
        effective_spread="0.05",
        quoted_spread="0.20",
    )

    cat = build_category_report([r1], OrderSizeBucket.SHARES_100_TO_499)

    assert cat.order_size_bucket == OrderSizeBucket.SHARES_100_TO_499
    assert cat.order_count == 1
    assert cat.executed_order_count == 1
    assert cat.total_order_quantity == Decimal("200")
    assert cat.total_executed_quantity == Decimal("200")
    assert cat.price_improvement == Decimal("0.10")
    assert cat.effective_spread == Decimal("0.05")
    assert cat.quoted_spread == Decimal("0.20")


def test_build_category_report_multiple_reports_same_bucket():
    r1 = make_report("R1", requested_qty="100", executed_qty="100", effective_spread="0.10")
    r2 = make_report("R2", requested_qty="300", executed_qty="300", effective_spread="0.20")

    cat = build_category_report([r1, r2], OrderSizeBucket.SHARES_100_TO_499)

    assert cat.order_count == 2
    assert cat.executed_order_count == 2
    assert cat.total_order_quantity == Decimal("400")
    assert cat.total_executed_quantity == Decimal("400")
    # ES: (100*0.10 + 300*0.20)/400 = 70/400 = 0.175
    assert cat.effective_spread == Decimal("0.175")


def test_build_category_report_excludes_different_buckets_and_non_reportable():
    r1 = make_report("R1", size_bucket=OrderSizeBucket.SHARES_100_TO_499, reportable=True)
    r2 = make_report("R2", size_bucket=OrderSizeBucket.SHARES_100_TO_499, reportable=False)
    r3 = make_report("R3", size_bucket=OrderSizeBucket.SHARES_500_TO_1999, reportable=True)

    cat = build_category_report([r1, r2, r3], OrderSizeBucket.SHARES_100_TO_499)

    assert cat.order_count == 1
    assert cat.total_order_quantity == Decimal("100")


def test_unexecuted_report_contributes_to_counts_not_execution_volume_statistics():
    r1 = make_report("R1", requested_qty="100", executed_qty="100", effective_spread="0.10")
    r2 = make_report("R2", requested_qty="100", executed_qty="0", effective_spread=None)  # unexecuted
    r3 = make_report("R3", requested_qty="100", executed_qty="50", effective_spread="0.20")  # partial

    cat = build_category_report([r1, r2, r3], OrderSizeBucket.SHARES_100_TO_499)

    assert cat.order_count == 3
    assert cat.executed_order_count == 2
    assert cat.total_order_quantity == Decimal("300")
    assert cat.total_executed_quantity == Decimal("150")
    # ES weighted by executed shares (100 and 50): (100*0.10 + 50*0.20)/150 = 20/150 = 0.1333333333333333333333333333
    assert cat.effective_spread == (Decimal("100") * Decimal("0.10") + Decimal("50") * Decimal("0.20")) / Decimal("150")


def test_none_metric_handling():
    r1 = make_report("R1", executed_qty="100", effective_spread="0.10")
    r2 = make_report("R2", executed_qty="200", effective_spread=None)

    cat = build_category_report([r1, r2], OrderSizeBucket.SHARES_100_TO_499)

    assert cat.order_count == 2
    assert cat.executed_order_count == 2
    assert cat.total_executed_quantity == Decimal("300")
    # ES: only R1 contributes -> (100*0.10)/100 = 0.10
    assert cat.effective_spread == Decimal("0.10")


def test_realized_spread_horizons_aggregate_independently():
    r1 = make_report(
        "R1",
        executed_qty="100",
        realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.10"),
            RealizedSpreadHorizon.S_1: Decimal("0.08"),
            RealizedSpreadHorizon.S_15: None,
            RealizedSpreadHorizon.M_1: Decimal("0.04"),
            RealizedSpreadHorizon.M_5: Decimal("0.01"),
        },
    )

    cat = build_category_report([r1], OrderSizeBucket.SHARES_100_TO_499)

    assert cat.realized_spreads[RealizedSpreadHorizon.MS_50] == Decimal("0.10")
    assert cat.realized_spreads[RealizedSpreadHorizon.S_1] == Decimal("0.08")
    assert cat.realized_spreads[RealizedSpreadHorizon.S_15] is None
    assert cat.realized_spreads[RealizedSpreadHorizon.M_1] == Decimal("0.04")
    assert cat.realized_spreads[RealizedSpreadHorizon.M_5] == Decimal("0.01")


def test_empty_category_report():
    cat = build_category_report([], OrderSizeBucket.ODD_LOT)

    assert cat.order_size_bucket == OrderSizeBucket.ODD_LOT
    assert cat.order_count == 0
    assert cat.executed_order_count == 0
    assert cat.total_order_quantity == Decimal("0")
    assert cat.total_executed_quantity == Decimal("0")
    assert cat.price_improvement is None
    assert cat.effective_spread is None
    assert cat.quoted_spread is None
    assert all(val is None for val in cat.realized_spreads.values())
