from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport


def filter_reportable_orders(
    reports: list[OrderReport],
) -> list[OrderReport]:
    return [report for report in reports if report.reportable]


def group_by_size_bucket(
    reports: list[OrderReport],
) -> dict[OrderSizeBucket, list[OrderReport]]:
    grouped: dict[OrderSizeBucket, list[OrderReport]] = {
        bucket: [] for bucket in OrderSizeBucket
    }

    for report in filter_reportable_orders(reports):
        grouped[report.order_size_bucket].append(report)

    return grouped
