from decimal import Decimal

from pydantic import BaseModel, Field

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


class CategoryReport(BaseModel):
    order_type_category: OrderTypeCategory
    order_size_bucket: OrderSizeBucket
    order_count: int = Field(ge=0)
    executed_order_count: int = Field(ge=0)
    total_order_quantity: Decimal = Field(ge=0)
    total_executed_quantity: Decimal = Field(ge=0)

    price_improvement: Decimal | None = None
    effective_spread: Decimal | None = None
    quoted_spread: Decimal | None = None

    realized_spreads: dict[
        RealizedSpreadHorizon,
        Decimal | None,
    ] = Field(default_factory=dict)

    percentage_effective_spread: Decimal | None = None
    percentage_quoted_spread: Decimal | None = None

    percentage_realized_spreads: dict[
        RealizedSpreadHorizon,
        Decimal | None,
    ] = Field(default_factory=dict)


def build_category_report(
    reports: list[OrderReport],
    category: OrderTypeCategory,
    bucket: OrderSizeBucket,
) -> CategoryReport:
    bucket_reports = [
        report
        for report in reports
        if (
            report.reportable
            and report.order_type_category == category
            and report.order_size_bucket == bucket
        )
    ]

    order_count = len(bucket_reports)
    executed_order_count = sum(
        1 for report in bucket_reports if report.executed_quantity > 0
    )

    total_order_quantity = sum(
        (report.requested_quantity for report in bucket_reports),
        Decimal("0"),
    )
    total_executed_quantity = sum(
        (report.executed_quantity for report in bucket_reports),
        Decimal("0"),
    )

    realized_spreads = {
        horizon: share_weighted_realized_spread(bucket_reports, horizon)
        for horizon in RealizedSpreadHorizon
    }

    percentage_realized_spreads = {
        horizon: share_weighted_percentage_realized_spread(bucket_reports, horizon)
        for horizon in RealizedSpreadHorizon
    }

    return CategoryReport(
        order_type_category=category,
        order_size_bucket=bucket,
        order_count=order_count,
        executed_order_count=executed_order_count,
        total_order_quantity=total_order_quantity,
        total_executed_quantity=total_executed_quantity,
        price_improvement=share_weighted_price_improvement(bucket_reports),
        effective_spread=share_weighted_effective_spread(bucket_reports),
        quoted_spread=share_weighted_quoted_spread(bucket_reports),
        realized_spreads=realized_spreads,
        percentage_effective_spread=share_weighted_percentage_effective_spread(bucket_reports),
        percentage_quoted_spread=share_weighted_percentage_quoted_spread(bucket_reports),
        percentage_realized_spreads=percentage_realized_spreads,
    )
