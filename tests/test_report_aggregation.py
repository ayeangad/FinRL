from decimal import Decimal

from finrl.rules.classification import OrderTypeCategory
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport
from finrl.rules.report_aggregation import (
    filter_reportable_orders,
    group_by_category_and_size_bucket,
    group_by_size_bucket,
)


def make_report(
    order_id: str,
    size_bucket: OrderSizeBucket,
    category: OrderTypeCategory = OrderTypeCategory.MARKET,
    reportable: bool = True,
    requested_qty: str = "100",
    executed_qty: str = "100",
) -> OrderReport:
    return OrderReport(
        order_id=order_id,
        order_size_bucket=size_bucket,
        order_type_category=category,
        reportable=reportable,
        requested_quantity=Decimal(requested_qty),
        executed_quantity=Decimal(executed_qty),
    )


def test_filter_reportable_orders_excludes_non_reportable():
    r1 = make_report("ORD-001", OrderSizeBucket.SHARES_100_TO_499, reportable=True)
    r2 = make_report("ORD-002", OrderSizeBucket.SHARES_100_TO_499, reportable=False)
    r3 = make_report("ORD-003", OrderSizeBucket.SHARES_500_TO_1999, reportable=True)

    result = filter_reportable_orders([r1, r2, r3])

    assert len(result) == 2
    assert r1 in result
    assert r3 in result
    assert r2 not in result


def test_group_by_size_bucket_empty_population():
    grouped = group_by_size_bucket([])

    assert set(grouped.keys()) == set(OrderSizeBucket)
    assert all(len(orders) == 0 for orders in grouped.values())


def test_group_by_category_and_size_bucket_empty_population():
    grouped = group_by_category_and_size_bucket([])

    assert len(grouped) == 30
    assert all(len(orders) == 0 for orders in grouped.values())


def test_group_by_category_and_size_bucket_groups_correctly():
    r1 = make_report("ORD-001", OrderSizeBucket.ODD_LOT, OrderTypeCategory.MARKET, reportable=True)
    r2 = make_report("ORD-002", OrderSizeBucket.SHARES_100_TO_499, OrderTypeCategory.MARKETABLE_LIMIT, reportable=True)
    r3 = make_report("ORD-003", OrderSizeBucket.SHARES_100_TO_499, OrderTypeCategory.MARKETABLE_LIMIT, reportable=True)
    r4 = make_report("ORD-004", OrderSizeBucket.SHARES_100_TO_499, OrderTypeCategory.MARKETABLE_LIMIT, reportable=False)
    r5 = make_report("ORD-005", OrderSizeBucket.SHARES_10000_PLUS, OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT, reportable=True)

    grouped = group_by_category_and_size_bucket([r1, r2, r3, r4, r5])

    assert len(grouped) == 30
    assert grouped[(OrderTypeCategory.MARKET, OrderSizeBucket.ODD_LOT)] == [r1]
    assert grouped[(OrderTypeCategory.MARKETABLE_LIMIT, OrderSizeBucket.SHARES_100_TO_499)] == [r2, r3]
    assert grouped[(OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT, OrderSizeBucket.SHARES_10000_PLUS)] == [r5]
    assert r4 not in grouped[(OrderTypeCategory.MARKETABLE_LIMIT, OrderSizeBucket.SHARES_100_TO_499)]
