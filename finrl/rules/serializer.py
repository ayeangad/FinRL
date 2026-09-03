import json
from decimal import Decimal

from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.rule_605_report import Rule605Report

PIPE_DELIMITED_HEADER = [
    "order_type_category",
    "order_size_bucket",
    "num_covered_orders",
    "num_executed_orders",
    "cumulative_shares",
    "cumulative_executed_shares",
    "shares_price_improved",
    "shares_at_quote",
    "shares_outside_quote",
    "price_improvement",
    "effective_spread",
    "quoted_spread",
    "realized_spread_50ms",
    "realized_spread_1s",
    "realized_spread_15s",
    "realized_spread_1m",
    "realized_spread_5m",
    "percentage_effective_spread",
    "percentage_quoted_spread",
    "percentage_realized_spread_50ms",
    "percentage_realized_spread_1s",
    "percentage_realized_spread_15s",
    "percentage_realized_spread_1m",
    "percentage_realized_spread_5m",
]


def _fmt(val: Decimal | int | str | None) -> str:
    if val is None:
        return ""
    return str(val)


def serialize_rule_605_pipe_delimited(report: Rule605Report) -> str:
    lines = ["|".join(PIPE_DELIMITED_HEADER)]

    for cell in report.all_cells():
        row = [
            _fmt(cell.order_type_category.value),
            _fmt(cell.order_size_bucket.value),
            _fmt(cell.num_covered_orders),
            _fmt(cell.num_executed_orders),
            _fmt(cell.cumulative_shares),
            _fmt(cell.cumulative_executed_shares),
            _fmt(cell.shares_price_improved),
            _fmt(cell.shares_at_quote),
            _fmt(cell.shares_outside_quote),
            _fmt(cell.price_improvement),
            _fmt(cell.effective_spread),
            _fmt(cell.quoted_spread),
            _fmt(cell.realized_spreads.get(RealizedSpreadHorizon.MS_50)),
            _fmt(cell.realized_spreads.get(RealizedSpreadHorizon.S_1)),
            _fmt(cell.realized_spreads.get(RealizedSpreadHorizon.S_15)),
            _fmt(cell.realized_spreads.get(RealizedSpreadHorizon.M_1)),
            _fmt(cell.realized_spreads.get(RealizedSpreadHorizon.M_5)),
            _fmt(cell.percentage_effective_spread),
            _fmt(cell.percentage_quoted_spread),
            _fmt(cell.percentage_realized_spreads.get(RealizedSpreadHorizon.MS_50)),
            _fmt(cell.percentage_realized_spreads.get(RealizedSpreadHorizon.S_1)),
            _fmt(cell.percentage_realized_spreads.get(RealizedSpreadHorizon.S_15)),
            _fmt(cell.percentage_realized_spreads.get(RealizedSpreadHorizon.M_1)),
            _fmt(cell.percentage_realized_spreads.get(RealizedSpreadHorizon.M_5)),
        ]
        lines.append("|".join(row))

    return "\n".join(lines)


def serialize_rule_605_json(report: Rule605Report) -> str:
    cells_json = []
    for cell in report.all_cells():
        cell_dict = cell.model_dump(mode="json")
        cells_json.append(cell_dict)
    return json.dumps(cells_json, indent=2)
