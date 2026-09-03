from finrl.rules.category_report import CategoryReport, build_category_report
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport
from finrl.rules.report_aggregation import (
    filter_reportable_orders,
    group_by_size_bucket,
)


def build_rule_605_report(
    reports: list[OrderReport],
) -> dict[OrderSizeBucket, CategoryReport]:
    reportable = filter_reportable_orders(reports)
    grouped = group_by_size_bucket(reportable)

    return {
        bucket: build_category_report(grouped[bucket], bucket)
        for bucket in OrderSizeBucket
    }
