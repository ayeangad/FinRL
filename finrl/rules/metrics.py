from decimal import Decimal

from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote

def midpoint(quote: Quote) -> Decimal:
    return (quote.bid_price + quote.ask_price) / Decimal("2")

def quoted_spread(quote: Quote) -> Decimal:
    return quote.ask_price - quote.bid_price

def price_improvement(
    side:OrderSide,
    execution_price: Decimal,
    quote: Quote,
) -> Decimal:
    if side == OrderSide.BUY:
        return quote.ask_price - execution_price
    
    return execution_price - quote.bid_price

def effective_spread(
    side: OrderSide,
    execution_price: Decimal,
    quote: Quote,
) -> Decimal:
    return Decimal("2") * abs(
        execution_price - midpoint(quote)
    )
