import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote


def load_scenario(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_order(data: dict, security: str) -> Order:
    return Order(
        order_id=data["order_id"],
        security=security,
        side=OrderSide(data["side"]),
        order_type=OrderType(data["order_type"]),
        quantity=Decimal(data["quantity"]),
        limit_price=(
            Decimal(data["limit_price"])
            if data.get("limit_price") is not None
            else None
        ),
        stop_price=(
            Decimal(data["stop_price"])
            if data.get("stop_price") is not None
            else None
        ),
        received_at=datetime.fromisoformat(data["received_at"]),
    )


def load_quotes(data: list[dict]) -> list[Quote]:
    return [
        Quote(
            security=quote["security"],
            bid_price=Decimal(quote["bid_price"]),
            bid_size=Decimal(quote["bid_size"]),
            ask_price=Decimal(quote["ask_price"]),
            ask_size=Decimal(quote["ask_size"]),
            timestamp=datetime.fromisoformat(
                quote["timestamp"]
            ),
        )
        for quote in data
    ]

def load_market(data: dict) -> MarketState:
    return MarketState(
        security=data["security"],
        quotes=load_quotes(data["quotes"]),
    )


def load_executions(data: list[dict]) -> list[Execution]:
    return [
        Execution(
            execution_id=execution["execution_id"],
            order_id=execution["order_id"],
            price=Decimal(execution["price"]),
            quantity=Decimal(execution["quantity"]),
            executed_at=datetime.fromisoformat(
                execution["executed_at"]
            ),
        )
        for execution in data
    ]

def validate_scenario(
    scenario: dict,
    executions: list[Execution],
) -> None:
    order_id = scenario["order"]["order_id"]
    requested_quantity = Decimal(scenario["order"]["quantity"])

    total_executed_quantity = sum(
        (execution.quantity for execution in executions),
        Decimal("0"),
    )

    for execution in executions:
        if execution.order_id != order_id:
            raise ValueError(
                f"Execution {execution.execution_id} references "
                f"order {execution.order_id}, expected {order_id}"
            )

    if total_executed_quantity > requested_quantity:
        raise ValueError(
            "Total executed quantity cannot exceed requested quantity"
        )