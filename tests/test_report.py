from decimal import Decimal

from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport


def test_create_order_report():
    report = OrderReport(
        order_id="ORD-0001",
        order_size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        reportable=True,
        requested_quantity=Decimal("100"),
        executed_quantity=Decimal("100"),
        average_execution_price=Decimal("100.12"),
        price_improvement=Decimal("0.10"),
        effective_spread=Decimal("0.10"),
        quoted_spread=Decimal("0.20"),
    )

    assert report.order_id == "ORD-0001"
    assert report.order_size_bucket == OrderSizeBucket.SHARES_100_TO_499
    assert report.reportable is True
    assert report.requested_quantity == Decimal("100")
    assert report.executed_quantity == Decimal("100")
    assert report.average_execution_price == Decimal("100.12")
    assert report.price_improvement == Decimal("0.10")
    assert report.effective_spread == Decimal("0.10")
    assert report.quoted_spread == Decimal("0.20")
    assert report.realized_spreads == {}


def test_order_report_allows_unexecuted_order():
    report = OrderReport(
        order_id="ORD-0002",
        order_size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        reportable=True,
        requested_quantity=Decimal("100"),
        executed_quantity=Decimal("0"),
        average_execution_price=None,
        price_improvement=None,
        effective_spread=None,
        quoted_spread=None,
    )

    assert report.order_size_bucket == OrderSizeBucket.SHARES_100_TO_499
    assert report.reportable is True
    assert report.executed_quantity == Decimal("0")
    assert report.average_execution_price is None
    assert report.price_improvement is None
    assert report.effective_spread is None
    assert report.quoted_spread is None
    assert report.realized_spreads == {}


def test_order_report_realized_spreads_explicit_values():
    report = OrderReport(
        order_id="ORD-0003",
        order_size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        reportable=True,
        requested_quantity=Decimal("100"),
        executed_quantity=Decimal("100"),
        average_execution_price=Decimal("100.00"),
        realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.10"),
            RealizedSpreadHorizon.S_1: None,
        },
    )

    assert report.order_size_bucket == OrderSizeBucket.SHARES_100_TO_499
    assert report.reportable is True
    assert report.realized_spreads[RealizedSpreadHorizon.MS_50] == Decimal("0.10")
    assert report.realized_spreads[RealizedSpreadHorizon.S_1] is None


