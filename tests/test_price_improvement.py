from datetime import datetime
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote
from finrl.rules.price_improvement import (
    PriceImprovementCategory,
    categorize_execution_shares,
    classify_execution_price_improvement,
)


def test_classify_execution_price_improvement_buy_order():
    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp=datetime.fromisoformat("2026-09-01T10:30:00.000"),
    )

    # Buy < ask (100.10) -> Price Improved
    assert (
        classify_execution_price_improvement(OrderSide.BUY, Decimal("100.05"), quote)
        == PriceImprovementCategory.PRICE_IMPROVED
    )

    # Buy == ask (100.10) -> At Quote
    assert (
        classify_execution_price_improvement(OrderSide.BUY, Decimal("100.10"), quote)
        == PriceImprovementCategory.AT_QUOTE
    )

    # Buy > ask (100.10) -> Outside Quote
    assert (
        classify_execution_price_improvement(OrderSide.BUY, Decimal("100.15"), quote)
        == PriceImprovementCategory.OUTSIDE_QUOTE
    )


def test_classify_execution_price_improvement_sell_order():
    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp=datetime.fromisoformat("2026-09-01T10:30:00.000"),
    )

    # Sell > bid (99.90) -> Price Improved
    assert (
        classify_execution_price_improvement(OrderSide.SELL, Decimal("99.95"), quote)
        == PriceImprovementCategory.PRICE_IMPROVED
    )

    # Sell == bid (99.90) -> At Quote
    assert (
        classify_execution_price_improvement(OrderSide.SELL, Decimal("99.90"), quote)
        == PriceImprovementCategory.AT_QUOTE
    )

    # Sell < bid (99.90) -> Outside Quote
    assert (
        classify_execution_price_improvement(OrderSide.SELL, Decimal("99.85"), quote)
        == PriceImprovementCategory.OUTSIDE_QUOTE
    )


def test_categorize_execution_shares_multiple_fills():
    now = datetime.fromisoformat("2026-09-01T10:30:00.000")
    quote = Quote(
        security="FINRL",
        bid_price=Decimal("99.90"),
        bid_size=Decimal("500"),
        ask_price=Decimal("100.10"),
        ask_size=Decimal("300"),
        timestamp=now,
    )

    executions = [
        Execution(execution_id="E1", order_id="O1", price=Decimal("100.00"), quantity=Decimal("100"), executed_at=now),
        Execution(execution_id="E2", order_id="O1", price=Decimal("100.10"), quantity=Decimal("200"), executed_at=now),
        Execution(execution_id="E3", order_id="O1", price=Decimal("100.20"), quantity=Decimal("50"), executed_at=now),
    ]

    improved, at_quote, outside = categorize_execution_shares(OrderSide.BUY, executions, quote)

    assert improved == Decimal("100")
    assert at_quote == Decimal("200")
    assert outside == Decimal("50")
