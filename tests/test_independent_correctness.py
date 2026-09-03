from datetime import datetime
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote
from finrl.evals.order_evaluator import evaluate_order
from finrl.rules.classification import OrderTypeCategory, classify_order_category
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.metrics import (
    effective_spread,
    midpoint,
    percentage_effective_spread,
    percentage_quoted_spread,
    percentage_realized_spread,
    price_improvement,
    quoted_spread,
    realized_spread,
)
from finrl.rules.order_size import OrderSizeBucket, classify_order_size
from finrl.rules.report_builder import build_rule_605_report


def test_hand_derived_metrics_primitives():
    # Hand-crafted quote: Bid 10.00, Ask 10.20 -> Midpoint = 10.10
    q = Quote(
        security="FINRL",
        bid_price=Decimal("10.00"),
        bid_size=Decimal("1000"),
        ask_price=Decimal("10.20"),
        ask_size=Decimal("1000"),
        timestamp=datetime.fromisoformat("2026-09-01T10:00:00"),
    )

    # 1. Midpoint & Quoted Spread
    assert midpoint(q) == Decimal("10.10")
    assert quoted_spread(q) == Decimal("0.20")
    assert percentage_quoted_spread(q) == Decimal("0.20") / Decimal("10.10")

    # 2. BUY side fill at 10.05 (Price improvement = Ask 10.20 - 10.05 = 0.15)
    pi = price_improvement(OrderSide.BUY, Decimal("10.05"), q)
    assert pi == Decimal("0.15")

    # 3. BUY side fill at 10.05 Effective spread = 2 * abs(10.05 - 10.10) = 0.10
    es = effective_spread(OrderSide.BUY, Decimal("10.05"), q)
    assert es == Decimal("0.10")
    pes = percentage_effective_spread(OrderSide.BUY, Decimal("10.05"), q)
    assert pes == Decimal("0.10") / Decimal("10.10")

    # 4. Realized spread at horizon mid = 10.20 -> BUY side RS = 2 * (10.05 - 10.20) = -0.30
    q_fut = Quote(
        security="FINRL",
        bid_price=Decimal("10.10"),
        bid_size=Decimal("1000"),
        ask_price=Decimal("10.30"),
        ask_size=Decimal("1000"),
        timestamp=datetime.fromisoformat("2026-09-01T10:00:00.050"),
    )
    rs = realized_spread(OrderSide.BUY, Decimal("10.05"), q_fut)
    assert rs == Decimal("-0.30")
    prs = percentage_realized_spread(OrderSide.BUY, Decimal("10.05"), q_fut, q)
    assert prs == Decimal("-0.30") / Decimal("10.10")


def test_hand_derived_classification_and_size_boundaries():
    q = Quote(
        security="FINRL",
        bid_price=Decimal("10.00"),
        bid_size=Decimal("1000"),
        ask_price=Decimal("10.20"),
        ask_size=Decimal("1000"),
        timestamp=datetime.fromisoformat("2026-09-01T10:00:00"),
    )

    # Size boundaries: 99 (odd lot), 100 (100-499), 499 (100-499), 500 (500-1999), 2000 (2000-4999), 5000 (5000-9999), 10000 (10000+)
    assert classify_order_size(Decimal("99")) == OrderSizeBucket.ODD_LOT
    assert classify_order_size(Decimal("100")) == OrderSizeBucket.SHARES_100_TO_499
    assert classify_order_size(Decimal("499")) == OrderSizeBucket.SHARES_100_TO_499
    assert classify_order_size(Decimal("500")) == OrderSizeBucket.SHARES_500_TO_1999
    assert classify_order_size(Decimal("2000")) == OrderSizeBucket.SHARES_2000_TO_4999
    assert classify_order_size(Decimal("5000")) == OrderSizeBucket.SHARES_5000_TO_9999
    assert classify_order_size(Decimal("10000")) == OrderSizeBucket.SHARES_10000_PLUS

    # 5-way Category Classification
    # BUY Market
    o_mkt = Order(order_id="1", security="FINRL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100"), received_at=q.timestamp)
    assert classify_order_category(o_mkt, q) == OrderTypeCategory.MARKET

    # BUY Limit >= Ask (10.20) -> Marketable Limit
    o_mkt_lim = Order(order_id="2", security="FINRL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("100"), limit_price=Decimal("10.20"), received_at=q.timestamp)
    assert classify_order_category(o_mkt_lim, q) == OrderTypeCategory.MARKETABLE_LIMIT

    # BUY Limit at Midpoint (10.10) -> Midpoint or Better Limit
    o_mid_lim = Order(order_id="3", security="FINRL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("100"), limit_price=Decimal("10.10"), received_at=q.timestamp)
    assert classify_order_category(o_mid_lim, q) == OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT

    # BUY Limit below Midpoint (10.05) -> Non-Marketable Limit
    o_non_mkt = Order(order_id="4", security="FINRL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("100"), limit_price=Decimal("10.05"), received_at=q.timestamp)
    assert classify_order_category(o_non_mkt, q) == OrderTypeCategory.NON_MARKETABLE_LIMIT

    # BUY Stop -> Stop
    o_stop = Order(order_id="5", security="FINRL", side=OrderSide.BUY, order_type=OrderType.STOP, quantity=Decimal("100"), stop_price=Decimal("10.30"), received_at=q.timestamp)
    assert classify_order_category(o_stop, q) == OrderTypeCategory.STOP


def test_hand_derived_multi_execution_end_to_end_scenario():
    # Setup independent scenario data
    t0 = datetime.fromisoformat("2026-09-01T10:00:00.000")
    q0 = Quote(security="FINRL", bid_price=Decimal("10.00"), bid_size=Decimal("1000"), ask_price=Decimal("10.20"), ask_size=Decimal("1000"), timestamp=t0)
    
    # Future quotes for realized spread horizons
    q_50ms = Quote(security="FINRL", bid_price=Decimal("10.10"), bid_size=Decimal("1000"), ask_price=Decimal("10.30"), ask_size=Decimal("1000"), timestamp=datetime.fromisoformat("2026-09-01T10:00:00.050"))
    q_1s = Quote(security="FINRL", bid_price=Decimal("10.00"), bid_size=Decimal("1000"), ask_price=Decimal("10.20"), ask_size=Decimal("1000"), timestamp=datetime.fromisoformat("2026-09-01T10:00:01.000"))
    q_15s = Quote(security="FINRL", bid_price=Decimal("9.90"), bid_size=Decimal("1000"), ask_price=Decimal("10.10"), ask_size=Decimal("1000"), timestamp=datetime.fromisoformat("2026-09-01T10:00:15.000"))
    q_1m = Quote(security="FINRL", bid_price=Decimal("9.80"), bid_size=Decimal("1000"), ask_price=Decimal("10.00"), ask_size=Decimal("1000"), timestamp=datetime.fromisoformat("2026-09-01T10:01:00.000"))
    q_5m = Quote(security="FINRL", bid_price=Decimal("9.70"), bid_size=Decimal("1000"), ask_price=Decimal("9.90"), ask_size=Decimal("1000"), timestamp=datetime.fromisoformat("2026-09-01T10:05:00.000"))

    market = MarketState(security="FINRL", quotes=[q0, q_50ms, q_1s, q_15s, q_1m, q_5m])

    # BUY Market 150 shares
    order = Order(order_id="ORD-HAND-1", security="FINRL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("150"), received_at=t0)
    
    # 2 Fills: 100 shares @ 10.05 (improved), 50 shares @ 10.20 (at quote)
    exec1 = Execution(execution_id="E1", order_id="ORD-HAND-1", price=Decimal("10.05"), quantity=Decimal("100"), executed_at=t0)
    exec2 = Execution(execution_id="E2", order_id="ORD-HAND-1", price=Decimal("10.20"), quantity=Decimal("50"), executed_at=t0)

    report = evaluate_order(order, [exec1, exec2], market)
    rule_605_report = build_rule_605_report([report])

    cell = rule_605_report.get_cell(OrderTypeCategory.MARKET, OrderSizeBucket.SHARES_100_TO_499)

    # Independent Expected Calculations:
    # 1. Counts & Shares
    assert cell.num_covered_orders == 1
    assert cell.num_executed_orders == 1
    assert cell.cumulative_shares == Decimal("150")
    assert cell.cumulative_executed_shares == Decimal("150")

    # 2. Price improvement breakdown
    # Fill 1 (100 sh @ 10.05 < Ask 10.20): Price Improved
    # Fill 2 (50 sh @ 10.20 == Ask 10.20): At Quote
    assert cell.shares_price_improved == Decimal("100")
    assert cell.shares_at_quote == Decimal("50")
    assert cell.shares_outside_quote == Decimal("0")

    # 3. Share-weighted Price Improvement
    # Fill 1 PI = 10.20 - 10.05 = 0.15 (100 sh)
    # Fill 2 PI = 10.20 - 10.20 = 0.00 (50 sh)
    # Weighted PI = (100 * 0.15 + 50 * 0.00) / 150 = 15 / 150 = 0.10
    assert cell.price_improvement == Decimal("0.10")

    # 4. Share-weighted Effective Spread
    # Fill 1 ES = 2 * abs(10.05 - 10.10) = 0.10 (100 sh)
    # Fill 2 ES = 2 * abs(10.20 - 10.10) = 0.20 (50 sh)
    # Weighted ES = (100 * 0.10 + 50 * 0.20) / 150 = 20 / 150 = 2/15
    assert cell.effective_spread == Decimal("2") / Decimal("15")

    # 5. Quoted Spread
    assert cell.quoted_spread == Decimal("0.20")

    # 6. Realized Spreads across 5 horizons (BUY side: 2 * (Fill_Price - Future_Midpoint))
    # Fill 1 (100 sh @ 10.05), Fill 2 (50 sh @ 10.20)
    # 50ms (Mid = 10.20): Fill 1 RS = 2 * (10.05 - 10.20) = -0.30 (100 sh); Fill 2 RS = 2 * (10.20 - 10.20) = 0.00 (50 sh)
    # Weighted 50ms RS = (100 * -0.30 + 50 * 0.00) / 150 = -30 / 150 = -0.20
    assert cell.realized_spreads[RealizedSpreadHorizon.MS_50] == Decimal("-0.20")
    
    # 1s (Mid = 10.10): Fill 1 RS = 2 * (10.05 - 10.10) = -0.10 (100 sh); Fill 2 RS = 2 * (10.20 - 10.10) = 0.20 (50 sh)
    # Weighted 1s RS = (100 * -0.10 + 50 * 0.20) / 150 = 0.00
    assert cell.realized_spreads[RealizedSpreadHorizon.S_1] == Decimal("0.00")

    # 15s (Mid = 10.00): Fill 1 RS = 2 * (10.05 - 10.00) = 0.10 (100 sh); Fill 2 RS = 2 * (10.20 - 10.00) = 0.40 (50 sh)
    # Weighted 15s RS = (100 * 0.10 + 50 * 0.40) / 150 = 30 / 150 = 0.20
    assert cell.realized_spreads[RealizedSpreadHorizon.S_15] == Decimal("0.20")

    # 1m (Mid = 9.90): Fill 1 RS = 2 * (10.05 - 9.90) = 0.30 (100 sh); Fill 2 RS = 2 * (10.20 - 9.90) = 0.60 (50 sh)
    # Weighted 1m RS = (100 * 0.30 + 50 * 0.60) / 150 = 60 / 150 = 0.40
    assert cell.realized_spreads[RealizedSpreadHorizon.M_1] == Decimal("0.40")

    # 5m (Mid = 9.80): Fill 1 RS = 2 * (10.05 - 9.80) = 0.50 (100 sh); Fill 2 RS = 2 * (10.20 - 9.80) = 0.80 (50 sh)
    # Weighted 5m RS = (100 * 0.50 + 50 * 0.80) / 150 = 90 / 150 = 0.60
    assert cell.realized_spreads[RealizedSpreadHorizon.M_5] == Decimal("0.60")

    # 7. Percentage Spread Metrics (Denominator = Receipt Midpoint = 10.10)
    assert cell.percentage_effective_spread == (Decimal("2") / Decimal("15")) / Decimal("10.10")
    assert cell.percentage_quoted_spread == Decimal("0.20") / Decimal("10.10")
    assert cell.percentage_realized_spreads[RealizedSpreadHorizon.MS_50] == Decimal("-0.20") / Decimal("10.10")
    assert cell.percentage_realized_spreads[RealizedSpreadHorizon.S_15] == Decimal("0.20") / Decimal("10.10")
