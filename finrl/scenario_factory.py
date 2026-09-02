from datetime import datetime
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.order import OrderSide, OrderType
from finrl.domain.quote import Quote
from finrl.scenarios import validate_scenario

def make_limit_order_scenario(
    *,
    scenario_id: str,
    security: str,
    side: OrderSide,
    quantity: Decimal,
    limit_price: Decimal,
    bid_price: Decimal,
    ask_price: Decimal,
    execution_price: Decimal,
) -> dict:
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")
    executed_at = datetime.fromisoformat("2026-09-01T10:30:00.200")
    
    scenario = {
        "scenario_id" : scenario_id,
        "version" : "v0.1",
        "security" : security,
        "order" : {
            "order_id" : f"{scenario_id}-ORDER",
            "side" : side.value,
            "order_type" : OrderType.LIMIT.value,
            "quantity":str(quantity),
            "limit_price":str(limit_price),
            "received_at": received_at.isoformat(),
        },
        "quotes" : [
            {
                "security" : security,
                "bid_price" : str(bid_price),
                "bid_size" : str(quantity),
                "ask_price" : str(ask_price),
                "ask_size" : str(quantity),
                "timestamp" : received_at.isoformat(),
            }
        ]
        ,
        "executions": [
            {
                "execution_id": f"{scenario_id}-EXECUTION",
                "order_id": f"{scenario_id}-ORDER",
                "price": str(execution_price),
                "quantity": str(quantity),
                "executed_at": executed_at.isoformat(),
            }
        ],
        "expected": {},
    }

    validate_scenario(scenario)

    return scenario
