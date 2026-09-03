from datetime import datetime, timedelta
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote
from finrl.evals.order_evaluator import evaluate_order
from finrl.rules.horizons import RealizedSpreadHorizon


def test_evaluate_partially_executed_buy_order():
    received_at = datetime.fromisoformat(
        "2026-09-01T10:30:00.100"
    )

    order = Order(
        order_id="ORD-0001",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.10"),
        received_at=received_at,
    )

    executions = [
        Execution(
            execution_id="EXE-0001",
            order_id="ORD-0001",
            price=Decimal("99.80"),
            quantity=Decimal("40"),
            executed_at=datetime.fromisoformat(
                "2026-09-01T10:30:00.200"
            ),
        ),
        Execution(
            execution_id="EXE-0002",
            order_id="ORD-0001",
            price=Decimal("100.20"),
            quantity=Decimal("20"),
            executed_at=datetime.fromisoformat(
                "2026-09-01T10:30:00.300"
            ),
        ),
    ]

    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            )
        ],
    )

    report = evaluate_order(
        order,
        executions,
        market,
    )

    assert report.order_id == "ORD-0001"
    assert report.requested_quantity == Decimal("100")
    assert report.executed_quantity == Decimal("60")

    # (40 * 99.80 + 20 * 100.20) / 60
    assert report.average_execution_price == (
        Decimal("99.80") * Decimal("40")
        + Decimal("100.20") * Decimal("20")
    ) / Decimal("60")

    assert report.price_improvement == (
        Decimal("40") * Decimal("0.30")
        + Decimal("20") * Decimal("-0.10")
    ) / Decimal("60")

    assert report.effective_spread == Decimal("0.40")

    assert report.quoted_spread == Decimal("0.20")


def test_evaluate_order_uses_quote_at_order_receipt():
    received_at = datetime.fromisoformat(
        "2026-09-01T10:30:00.100"
    )
    execution_at = datetime.fromisoformat(
        "2026-09-01T10:30:00.200"
    )

    order = Order(
        order_id="ORD-0002",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.10"),
        received_at=received_at,
    )

    executions = [
        Execution(
            execution_id="EXE-0003",
            order_id="ORD-0002",
            price=Decimal("100.00"),
            quantity=Decimal("100"),
            executed_at=execution_at,
        )
    ]

    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.80"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat(
                    "2026-09-01T10:30:00.150"
                ),
            ),
        ],
    )

    report = evaluate_order(
        order,
        executions,
        market,
    )

    assert report.price_improvement == Decimal("0.10")
    assert report.quoted_spread == Decimal("0.20")
    assert report.effective_spread == Decimal("0.00")


def test_evaluate_partially_executed_sell_order():
    received_at = datetime.fromisoformat(
        "2026-09-01T10:30:00.100"
    )

    order = Order(
        order_id="ORD-0003",
        security="FINRL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("99.90"),
        received_at=received_at,
    )

    executions = [
        Execution(
            execution_id="EXE-0004",
            order_id="ORD-0003",
            price=Decimal("100.20"),
            quantity=Decimal("40"),
            executed_at=datetime.fromisoformat(
                "2026-09-01T10:30:00.200"
            ),
        ),
        Execution(
            execution_id="EXE-0005",
            order_id="ORD-0003",
            price=Decimal("99.80"),
            quantity=Decimal("20"),
            executed_at=datetime.fromisoformat(
                "2026-09-01T10:30:00.300"
            ),
        ),
    ]

    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            )
        ],
    )

    report = evaluate_order(
        order,
        executions,
        market,
    )

    assert report.executed_quantity == Decimal("60")

    assert report.average_execution_price == (
        Decimal("100.20") * Decimal("40")
        + Decimal("99.80") * Decimal("20")
    ) / Decimal("60")

    assert report.price_improvement == (
        Decimal("40") * Decimal("0.30")
        + Decimal("20") * Decimal("-0.10")
    ) / Decimal("60")

    assert report.effective_spread == (
        Decimal("40") * Decimal("0.40")
        + Decimal("20") * Decimal("0.40")
    ) / Decimal("60")

    assert report.quoted_spread == Decimal("0.20")


def test_evaluate_order_per_execution_per_horizon_realized_spread():
    received_at = datetime.fromisoformat(
        "2026-09-01T10:30:00.000"
    )

    order = Order(
        order_id="ORD-0004",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.10"),
        received_at=received_at,
    )

    # EX1: executed at 10:30:00.100, price 100.00, qty 40
    # EX2: executed at 10:30:00.200, price 100.30, qty 20
    ex1 = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0004",
        price=Decimal("100.00"),
        quantity=Decimal("40"),
        executed_at=datetime.fromisoformat("2026-09-01T10:30:00.100"),
    )
    ex2 = Execution(
        execution_id="EXE-0002",
        order_id="ORD-0004",
        price=Decimal("100.30"),
        quantity=Decimal("20"),
        executed_at=datetime.fromisoformat("2026-09-01T10:30:00.200"),
    )

    # Receipt Quote: 10:30:00.000 -> 99.90 / 100.10 (midpoint 100.00)
    # 50ms horizon:
    #   EX1 target: 10:30:00.150 -> quote at 10:30:00.150: 100.10 / 100.30 (midpoint 100.20)
    #   EX2 target: 10:30:00.250 -> quote at 10:30:00.250: 100.00 / 100.20 (midpoint 100.10)
    #   EX1 RS = 2 * (100.00 - 100.20) = -0.40
    #   EX2 RS = 2 * (100.30 - 100.10) = +0.40
    #   50ms share-weighted RS = (40 * -0.40 + 20 * 0.40) / 60 = -0.133333...

    # 1s horizon:
    #   EX1 target: 10:30:01.100 -> quote at 10:30:01.100: 99.80 / 100.00 (midpoint 99.90)
    #   EX2 target: 10:30:01.200 -> quote at 10:30:01.200: 99.80 / 100.00 (midpoint 99.90)
    #   EX1 RS = 2 * (100.00 - 99.90) = +0.20
    #   EX2 RS = 2 * (100.30 - 99.90) = +0.80
    #   1s share-weighted RS = (40 * 0.20 + 20 * 0.80) / 60 = +0.40

    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("100.10"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.30"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.150"),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("100.00"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.20"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.250"),
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.80"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.00"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:01.100"),
            ),
        ],
    )

    report = evaluate_order(
        order,
        [ex1, ex2],
        market,
    )

    rs_50ms = report.realized_spreads[RealizedSpreadHorizon.MS_50]
    rs_1s = report.realized_spreads[RealizedSpreadHorizon.S_1]

    assert rs_50ms == (Decimal("40") * Decimal("-0.40") + Decimal("20") * Decimal("0.40")) / Decimal("60")
    assert rs_1s == (Decimal("40") * Decimal("0.20") + Decimal("20") * Decimal("0.80")) / Decimal("60")
    assert rs_50ms != rs_1s


def test_evaluate_order_realized_spread_partial_missing_horizon():
    received_at = datetime.fromisoformat(
        "2026-09-01T10:30:00.000"
    )

    order = Order(
        order_id="ORD-0005",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("100.10"),
        received_at=received_at,
    )

    ex = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0005",
        price=Decimal("100.00"),
        quantity=Decimal("100"),
        executed_at=datetime.fromisoformat("2026-09-01T10:30:00.100"),
    )

    # Market has quote at receipt (0.000) and at 50ms horizon (0.150).
    # But market sequence ends at 10:30:02.000 (valid for 50ms, 1s, 15s).
    # No quote exists for 1m (10:31:00.100) or 5m (10:35:00.100).
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("100.10"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.30"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.150"),
            ),
        ],
    )

    report = evaluate_order(
        order,
        [ex],
        market,
    )

    # 50ms target (10:30:00.150) -> quote at 0.150 (midpoint 100.20) -> RS = 2*(100.00 - 100.20) = -0.40
    # 1s target (10:30:01.100) -> quote at 0.150 (midpoint 100.20) -> RS = -0.40
    # 15s target (10:30:15.100) -> quote at 0.150 (midpoint 100.20) -> RS = -0.40
    assert report.realized_spreads[RealizedSpreadHorizon.MS_50] == Decimal("-0.40")
    assert report.realized_spreads[RealizedSpreadHorizon.S_1] == Decimal("-0.40")
    assert report.realized_spreads[RealizedSpreadHorizon.S_15] == Decimal("-0.40")


def test_evaluate_reportable_market_order():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")
    order = Order(
        order_id="ORD-MKT-01",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("250"),
        received_at=received_at,
    )
    executions = [
        Execution(
            execution_id="EXE-MKT-01",
            order_id="ORD-MKT-01",
            price=Decimal("100.00"),
            quantity=Decimal("250"),
            executed_at=datetime.fromisoformat("2026-09-01T10:30:00.150"),
        )
    ]
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            )
        ],
    )

    report = evaluate_order(order, executions, market)

    from finrl.rules.order_size import OrderSizeBucket

    assert report.reportable is True
    assert report.order_size_bucket == OrderSizeBucket.SHARES_100_TO_499
    assert report.executed_quantity == Decimal("250")
    assert report.price_improvement is not None
    assert report.effective_spread is not None


def test_evaluate_non_marketable_limit_never_executable():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")
    # BUY limit 99.50 (below receipt bid 99.90, market never drops to 99.50)
    order = Order(
        order_id="ORD-NML-01",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("300"),
        limit_price=Decimal("99.50"),
        received_at=received_at,
    )
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.90"),
                bid_size=Decimal("500"),
                ask_price=Decimal("100.10"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            )
        ],
    )

    report = evaluate_order(order, [], market)

    from finrl.rules.order_size import OrderSizeBucket

    assert report.reportable is False
    assert report.order_size_bucket == OrderSizeBucket.SHARES_100_TO_499
    assert report.executed_quantity == Decimal("0")
    assert report.average_execution_price is None
    assert report.price_improvement is None
    assert report.effective_spread is None
    assert report.quoted_spread is None
    assert report.realized_spreads[RealizedSpreadHorizon.MS_50] is None


def test_evaluate_non_marketable_limit_executable_unexecuted():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")
    # BUY limit 99.60 (below receipt bid 99.70, but market drops to bid 99.60 at T+100ms)
    order = Order(
        order_id="ORD-NML-02",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1500"),
        limit_price=Decimal("99.60"),
        received_at=received_at,
    )
    market = MarketState(
        security="FINRL",
        quotes=[
            Quote(
                security="FINRL",
                bid_price=Decimal("99.70"),
                bid_size=Decimal("500"),
                ask_price=Decimal("99.90"),
                ask_size=Decimal("300"),
                timestamp=received_at,
            ),
            Quote(
                security="FINRL",
                bid_price=Decimal("99.60"),
                bid_size=Decimal("500"),
                ask_price=Decimal("99.80"),
                ask_size=Decimal("300"),
                timestamp=datetime.fromisoformat("2026-09-01T10:30:00.200"),
            ),
        ],
    )

    # 0 executions (order became executable, but received 0 fills)
    report = evaluate_order(order, [], market)

    from finrl.rules.order_size import OrderSizeBucket

    assert report.reportable is True
    assert report.order_size_bucket == OrderSizeBucket.SHARES_500_TO_1999
    assert report.executed_quantity == Decimal("0")
    assert report.average_execution_price is None
    assert report.price_improvement is None
    assert report.effective_spread is None
    assert report.quoted_spread is None
    assert report.realized_spreads[RealizedSpreadHorizon.MS_50] is None


def test_evaluate_order_assigns_correct_order_type_category():
    from finrl.rules.classification import OrderTypeCategory

    received_at = datetime.fromisoformat("2026-09-01T10:30:00.000")
    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp=received_at,
    )
    market = MarketState(security="FINRL", quotes=[quote])

    # 1. Market order
    mkt = Order(order_id="O1", security="FINRL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100"), received_at=received_at)
    r_mkt = evaluate_order(mkt, [], market)
    assert r_mkt.order_type_category == OrderTypeCategory.MARKET

    # 2. Marketable Limit order (BUY @ 100.10)
    mkt_lim = Order(order_id="O2", security="FINRL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("100"), limit_price=Decimal("100.10"), received_at=received_at)
    r_mkt_lim = evaluate_order(mkt_lim, [], market)
    assert r_mkt_lim.order_type_category == OrderTypeCategory.MARKETABLE_LIMIT

    # 3. Midpoint-or-better Limit order (BUY @ 100.00)
    mid_lim = Order(order_id="O3", security="FINRL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("100"), limit_price=Decimal("100.00"), received_at=received_at)
    r_mid_lim = evaluate_order(mid_lim, [], market)
    assert r_mid_lim.order_type_category == OrderTypeCategory.MIDPOINT_OR_BETTER_LIMIT

    # 4. Non-marketable Limit order (BUY @ 99.95)
    non_mkt_lim = Order(order_id="O4", security="FINRL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("100"), limit_price=Decimal("99.95"), received_at=received_at)
    r_non_mkt_lim = evaluate_order(non_mkt_lim, [], market)
    assert r_non_mkt_lim.order_type_category == OrderTypeCategory.NON_MARKETABLE_LIMIT

    # 5. Stop order
    stop_ord = Order(order_id="O5", security="FINRL", side=OrderSide.BUY, order_type=OrderType.STOP, quantity=Decimal("100"), stop_price=Decimal("100.20"), received_at=received_at)
    r_stop = evaluate_order(stop_ord, [], market)
    assert r_stop.order_type_category == OrderTypeCategory.STOP


def test_evaluate_order_percentage_metrics_executed_order():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.000")
    receipt_quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp=received_at,
    )  # midpoint = 100.00

    # 50ms quote: midpoint = 100.20
    q_50ms = Quote(
        security="FINRL",
        bid_price=Decimal("100.10"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.30"),
        ask_size=Decimal("300"),
        timestamp=received_at + timedelta(milliseconds=50),
    )

    market = MarketState(security="FINRL", quotes=[receipt_quote, q_50ms])

    order = Order(
        order_id="ORD-PCT-01",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        received_at=received_at,
    )

    executions = [
        Execution(execution_id="E1", order_id="ORD-PCT-01", price=Decimal("100.05"), quantity=Decimal("40"), executed_at=received_at),
        Execution(execution_id="E2", order_id="ORD-PCT-01", price=Decimal("100.15"), quantity=Decimal("60"), executed_at=received_at),
    ]

    report = evaluate_order(order, executions, market)

    assert report.reportable is True
    # Quoted % = 0.20 / 100.00 = 0.002
    assert report.percentage_quoted_spread == Decimal("0.002")

    # ES1 % = 2*|100.05 - 100.00| / 100.00 = 0.10 / 100.00 = 0.001
    # ES2 % = 2*|100.15 - 100.00| / 100.00 = 0.30 / 100.00 = 0.003
    # Weighted %ES = (40*0.001 + 60*0.003)/100 = (0.04 + 0.18)/100 = 0.0022
    assert report.percentage_effective_spread == Decimal("0.0022")

    # RS1 % 50ms = 2*(100.05 - 100.20) / 100.00 = -0.30 / 100.00 = -0.003
    # RS2 % 50ms = 2*(100.15 - 100.20) / 100.00 = -0.10 / 100.00 = -0.001
    # Weighted %RS 50ms = (40*-0.003 + 60*-0.001)/100 = -0.0018
    assert report.percentage_realized_spreads[RealizedSpreadHorizon.MS_50] == Decimal("-0.0018")


def test_evaluate_order_percentage_metrics_missing_horizon_quote():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.000")
    receipt_quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp=received_at,
    )
    # Market only has the receipt quote at received_at - 1 minute (so for execution at received_at - 1m, 50ms quote is absent if market quotes start after execution + 50ms)
    # If execution occurs BEFORE any quotes in market:
    market = MarketState(security="FINRL", quotes=[receipt_quote])

    order = Order(
        order_id="ORD-PCT-MISS",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        received_at=received_at,
    )

    # Execution happened 10 minutes BEFORE receipt_quote was published
    earlier_exec = received_at - timedelta(minutes=10)
    executions = [
        Execution(execution_id="E1", order_id="ORD-PCT-MISS", price=Decimal("100.00"), quantity=Decimal("100"), executed_at=earlier_exec),
    ]

    report = evaluate_order(order, executions, market)
    assert report.percentage_realized_spreads[RealizedSpreadHorizon.MS_50] is None



def test_evaluate_order_percentage_metrics_unexecuted_or_non_reportable():
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.000")
    receipt_quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp=received_at,
    )
    market = MarketState(security="FINRL", quotes=[receipt_quote])

    order = Order(
        order_id="ORD-PCT-02",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        received_at=received_at,
    )

    # 0 executions (unexecuted)
    r_unexec = evaluate_order(order, [], market)
    assert r_unexec.percentage_effective_spread is None
    assert r_unexec.percentage_quoted_spread is None
    assert all(v is None for v in r_unexec.percentage_realized_spreads.values())

    # Non-reportable order (received outside market hours)
    out_of_hours = datetime.fromisoformat("2026-09-01T08:00:00.000")
    order_non_rep = Order(
        order_id="ORD-PCT-03",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        received_at=out_of_hours,
    )
    r_non_rep = evaluate_order(order_non_rep, [], market)
    assert r_non_rep.percentage_effective_spread is None
    assert r_non_rep.percentage_quoted_spread is None
    assert all(v is None for v in r_non_rep.percentage_realized_spreads.values())