from datetime import datetime
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote
from finrl.evals.order_evaluator import evaluate_order


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
            price=Decimal("100.00"),
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

    # (40 * 100.00 + 20 * 100.20) / 60
    assert report.average_execution_price == (
        Decimal("100.00") * Decimal("40")
        + Decimal("100.20") * Decimal("20")
    ) / Decimal("60")

    assert report.price_improvement == (
    Decimal("100.10") - report.average_execution_price
)

    assert report.effective_spread == (
    Decimal("2")
    * abs(
        report.average_execution_price
        - Decimal("100.00")
    )
)

    assert report.quoted_spread == Decimal("0.20")
