import json
from decimal import Decimal

from finrl.rules.classification import OrderTypeCategory
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report import OrderReport
from finrl.rules.report_builder import build_rule_605_report
from finrl.rules.serializer import (
    PIPE_DELIMITED_HEADER,
    serialize_rule_605_json,
    serialize_rule_605_pipe_delimited,
)


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
    percentage_effective_spread: str | None = None,
    percentage_quoted_spread: str | None = None,
    percentage_realized_spreads: dict[RealizedSpreadHorizon, Decimal | None] | None = None,
    shares_price_improved: str = "0",
    shares_at_quote: str = "0",
    shares_outside_quote: str = "0",
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
        percentage_effective_spread=(
            Decimal(percentage_effective_spread)
            if percentage_effective_spread is not None
            else None
        ),
        percentage_quoted_spread=(
            Decimal(percentage_quoted_spread)
            if percentage_quoted_spread is not None
            else None
        ),
        percentage_realized_spreads=percentage_realized_spreads or {},
        shares_price_improved=Decimal(shares_price_improved),
        shares_at_quote=Decimal(shares_at_quote),
        shares_outside_quote=Decimal(shares_outside_quote),
    )


def test_serialize_rule_605_pipe_delimited_golden_output():
    r1 = make_report(
        "ORD-GOLD-1",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MARKET,
        requested_qty="100",
        executed_qty="100",
        price_improvement="0.05",
        effective_spread="0.10",
        quoted_spread="0.20",
        percentage_effective_spread="0.0010",
        percentage_quoted_spread="0.0020",
        realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.08"),
        },
        percentage_realized_spreads={
            RealizedSpreadHorizon.MS_50: Decimal("0.0008"),
        },
        shares_price_improved="60",
        shares_at_quote="40",
    )

    report = build_rule_605_report([r1])
    pipe_output = serialize_rule_605_pipe_delimited(report)

    lines = pipe_output.strip().split("\n")
    # 1 header + 30 cells = 31 lines
    assert len(lines) == 31
    assert lines[0] == "|".join(PIPE_DELIMITED_HEADER)

    # Check populated MARKET x 100-499 line
    market_100_line = [
        line for line in lines[1:] if line.startswith("market|100_to_499")
    ]
    assert len(market_100_line) == 1
    fields = market_100_line[0].split("|")

    assert fields[0] == "market"
    assert fields[1] == "100_to_499"
    assert fields[2] == "1"  # num_covered_orders
    assert fields[3] == "1"  # num_executed_orders
    assert fields[4] == "100"  # cumulative_shares
    assert fields[5] == "100"  # cumulative_executed_shares
    assert fields[6] == "60"  # shares_price_improved
    assert fields[7] == "40"  # shares_at_quote
    assert fields[8] == "0"  # shares_outside_quote
    assert fields[9] == "0.05"  # price_improvement
    assert fields[10] == "0.10"  # effective_spread
    assert fields[11] == "0.20"  # quoted_spread
    assert fields[12] == "0.08"  # realized_spread_50ms
    assert fields[13] == ""  # realized_spread_1s (None)
    assert fields[17] == "0.0010"  # percentage_effective_spread
    assert fields[18] == "0.0020"  # percentage_quoted_spread
    assert fields[19] == "0.0008"  # percentage_realized_spread_50ms


def test_serialize_rule_605_json_output():
    r1 = make_report(
        "ORD-GOLD-2",
        size_bucket=OrderSizeBucket.SHARES_100_TO_499,
        category=OrderTypeCategory.MARKET,
        requested_qty="100",
        executed_qty="100",
    )

    report = build_rule_605_report([r1])
    json_output = serialize_rule_605_json(report)

    data = json.loads(json_output)
    assert isinstance(data, list)
    assert len(data) == 30
    assert data[0]["order_type_category"] in [cat.value for cat in OrderTypeCategory]
