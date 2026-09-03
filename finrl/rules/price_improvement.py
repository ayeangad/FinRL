from decimal import Decimal
from enum import Enum

from finrl.domain.execution import Execution
from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote


class PriceImprovementCategory(str, Enum):
    PRICE_IMPROVED = "price_improved"
    AT_QUOTE = "at_quote"
    OUTSIDE_QUOTE = "outside_quote"


def classify_execution_price_improvement(
    side: OrderSide,
    execution_price: Decimal,
    receipt_quote: Quote,
) -> PriceImprovementCategory:
    if side == OrderSide.BUY:
        if execution_price < receipt_quote.ask_price:
            return PriceImprovementCategory.PRICE_IMPROVED
        elif execution_price == receipt_quote.ask_price:
            return PriceImprovementCategory.AT_QUOTE
        else:
            return PriceImprovementCategory.OUTSIDE_QUOTE
    else:
        if execution_price > receipt_quote.bid_price:
            return PriceImprovementCategory.PRICE_IMPROVED
        elif execution_price == receipt_quote.bid_price:
            return PriceImprovementCategory.AT_QUOTE
        else:
            return PriceImprovementCategory.OUTSIDE_QUOTE


def categorize_execution_shares(
    side: OrderSide,
    executions: list[Execution],
    receipt_quote: Quote,
) -> tuple[Decimal, Decimal, Decimal]:
    """Returns (shares_price_improved, shares_at_quote, shares_outside_quote)."""
    improved = Decimal("0")
    at_quote = Decimal("0")
    outside = Decimal("0")

    for ex in executions:
        cat = classify_execution_price_improvement(side, ex.price, receipt_quote)
        if cat == PriceImprovementCategory.PRICE_IMPROVED:
            improved += ex.quantity
        elif cat == PriceImprovementCategory.AT_QUOTE:
            at_quote += ex.quantity
        else:
            outside += ex.quantity

    return improved, at_quote, outside
