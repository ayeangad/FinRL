from decimal import Decimal

from finrl.rules.aggregation import share_weighted_average
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.report import OrderReport


def share_weighted_price_improvement(
    reports: list[OrderReport],
) -> Decimal | None:
    values = [
        (report.price_improvement, report.executed_quantity)
        for report in reports
        if report.price_improvement is not None
    ]
    return share_weighted_average(values)


def share_weighted_effective_spread(
    reports: list[OrderReport],
) -> Decimal | None:
    values = [
        (report.effective_spread, report.executed_quantity)
        for report in reports
        if report.effective_spread is not None
    ]
    return share_weighted_average(values)


def share_weighted_quoted_spread(
    reports: list[OrderReport],
) -> Decimal | None:
    values = [
        (report.quoted_spread, report.executed_quantity)
        for report in reports
        if report.quoted_spread is not None
    ]
    return share_weighted_average(values)


def share_weighted_realized_spread(
    reports: list[OrderReport],
    horizon: RealizedSpreadHorizon,
) -> Decimal | None:
    values = []
    for report in reports:
        value = report.realized_spreads.get(horizon)
        if value is not None:
            values.append((value, report.executed_quantity))
    return share_weighted_average(values)


def share_weighted_percentage_effective_spread(
    reports: list[OrderReport],
) -> Decimal | None:
    values = [
        (report.percentage_effective_spread, report.executed_quantity)
        for report in reports
        if report.percentage_effective_spread is not None
    ]
    return share_weighted_average(values)


def share_weighted_percentage_quoted_spread(
    reports: list[OrderReport],
) -> Decimal | None:
    values = [
        (report.percentage_quoted_spread, report.executed_quantity)
        for report in reports
        if report.percentage_quoted_spread is not None
    ]
    return share_weighted_average(values)


def share_weighted_percentage_realized_spread(
    reports: list[OrderReport],
    horizon: RealizedSpreadHorizon,
) -> Decimal | None:
    values = []
    for report in reports:
        value = report.percentage_realized_spreads.get(horizon)
        if value is not None:
            values.append((value, report.executed_quantity))
    return share_weighted_average(values)


def sum_shares_price_improved(reports: list[OrderReport]) -> Decimal:
    return sum(
        (r.shares_price_improved for r in reports if r.reportable),
        Decimal(0),
    )


def sum_shares_at_quote(reports: list[OrderReport]) -> Decimal:
    return sum(
        (r.shares_at_quote for r in reports if r.reportable),
        Decimal(0),
    )


def sum_shares_outside_quote(reports: list[OrderReport]) -> Decimal:
    return sum(
        (r.shares_outside_quote for r in reports if r.reportable),
        Decimal(0),
    )
