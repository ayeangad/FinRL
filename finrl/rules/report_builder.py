from finrl.rules.category_report import build_category_report
from finrl.rules.classification import OrderTypeCategory
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport
from finrl.rules.report_aggregation import (
    filter_reportable_orders,
    group_by_category_and_size_bucket,
)
from finrl.rules.rule_605_report import Rule605Report


def build_rule_605_report(
    reports: list[OrderReport],
) -> Rule605Report:
    reportable = filter_reportable_orders(reports)
    grouped = group_by_category_and_size_bucket(reportable)

    categories = {
        (category, bucket): build_category_report(
            grouped[(category, bucket)],
            category,
            bucket,
        )
        for category in OrderTypeCategory
        for bucket in OrderSizeBucket
    }

    return Rule605Report(categories=categories)
