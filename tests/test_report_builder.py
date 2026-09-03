from decimal import Decimal

from finrl.rules.category_report import CategoryReport
from finrl.rules.classification import OrderTypeCategory
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport
from finrl.rules.report_builder import build_rule_605_report


def make_report(
    order_id: str,
    size_bucket: OrderSizeBucket = OrderSizeBucket.SHARES_100_TO_499,
    category: OrderTypeCategory = OrderTypeCategory.MARKET,
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
        order_type_category=category,
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


def test_build_rule_605_report_empty_input():
    result = build_rule_605_report([])

    assert len(result) == 30
    all_keys = {
        (cat, size)
        for cat in OrderTypeCategory
        for size in OrderSizeBucket
    }
    assert set(result.keys()) == all_keys

    for cat in result.values():
        assert isinstance(cat, CategoryReport)
        assert cat.order_count == 0
        assert cat.executed_order_count == 0
        assert cat.total_order_quantity == Decimal("0")
        assert cat.total_executed_quantity == Decimal("0")
        assert cat.price_improvement is None
        assert cat.effective_spread is None
        assert cat.quoted_spread is None
        assert all(v is None for v in cat.realized_spreads.values())


def test_build_rule_605_report_mixed_population():
    # Odd lot Market (reportable)
    r1 = make_report(
        "R1",
        size_bucket=OrderSizeBucket.ODD_LOT,
        category=OrderTypeCategory.MARKET,
        reportable=True,
        requested_qty="50",
        executed_qty="50",
        effective_spread="0.10",
    )
    # 100-499 Marketable Limit (reportable)
    r2 = make_report(
        "R2",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MARKETABLE_LIMIT,
        reportable=True,
        requested_qty="200",
        executed_qty="200",
        effective_spread="0.20",
    )
    # 100-499 Marketable Limit (non-reportable -> should be excluded)
    r3 = make_report(
        "R3",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MARKETABLE_LIMIT,
        reportable=False,
        requested_qty="300",
        executed_qty="0",
    )

    result = build_rule_605_report([r1, r2, r3])

    assert len(result) == 30

    cat_odd = result[(OrderTypeCategory.MARKET, OrderSizeBucket.ODD_LOT)]
    assert cat_odd.order_count == 1
    assert cat_odd.total_executed_quantity == Decimal("50")
    assert cat_odd.effective_spread == Decimal("0.10")

    cat_100 = result[(OrderTypeCategory.MARKETABLE_LIMIT, OrderSizeBucket.SHARES_100_TO_499)]
    assert cat_100.order_count == 1  # r3 excluded because reportable=False
    assert cat_100.total_executed_quantity == Decimal("200")
    assert cat_100.effective_spread == Decimal("0.20")

    # Remaining buckets empty
    assert result[(OrderTypeCategory.MARKET, OrderSizeBucket.SHARES_500_TO_1999)].order_count == 0


def test_build_rule_605_report_unexecuted_reportable_order():
    r1 = make_report(
        "R1",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MARKETABLE_LIMIT,
        reportable=True,
        requested_qty="100",
        executed_qty="100",
        effective_spread="0.10",
    )
    r2 = make_report(
        "R2",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MARKETABLE_LIMIT,
        reportable=True,
        requested_qty="100",
        executed_qty="0",  # unexecuted reportable order
    )

    result = build_rule_605_report([r1, r2])

    cat = result[(OrderTypeCategory.MARKETABLE_LIMIT, OrderSizeBucket.SHARES_100_TO_499)]
    assert cat.order_count == 2
    assert cat.executed_order_count == 1
    assert cat.total_order_quantity == Decimal("200")
    assert cat.total_executed_quantity == Decimal("100")
    assert cat.effective_spread == Decimal("0.10")


def test_build_rule_605_report_different_categories_land_in_different_cells():
    r1 = make_report(
        "R1",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MARKET,
        requested_qty="100",
        executed_qty="100",
        effective_spread="0.10",
    )
    r2 = make_report(
        "R2",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT,
        requested_qty="100",
        executed_qty="100",
        effective_spread="0.05",
    )

    result = build_rule_605_report([r1, r2])

    cat1 = result[(OrderTypeCategory.MARKET, OrderSizeBucket.SHARES_100_TO_499)]
    assert cat1.order_count == 1
    assert cat1.effective_spread == Decimal("0.10")

    cat2 = result[(OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT, OrderSizeBucket.SHARES_100_TO_499)]
    assert cat2.order_count == 1
    assert cat2.effective_spread == Decimal("0.05")
