from datetime import datetime
from decimal import Decimal

from finrl.domain.order import Order, OrderSide, OrderType

def test_create_order():
    order = Order(
        order_id="ORD-0001",
        security="FINRL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        limit_price=Decimal("99.99"),
        received_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.100"
        ),
    )

    assert order.order_id == "ORD-0001"
    assert order.side == OrderSide.BUY
    assert order.quantity == Decimal("100")
    assert order.limit_price == Decimal("99.99")