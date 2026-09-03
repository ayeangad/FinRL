from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order, OrderSide, OrderType
from finrl.domain.quote import Quote


class OrderInput(BaseModel):
    order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    received_at: datetime


class QuoteInput(BaseModel):
    security: str = "FINRL"
    bid_price: Decimal = Field(gt=0)
    bid_size: Decimal = Field(ge=0)
    ask_price: Decimal = Field(gt=0)
    ask_size: Decimal = Field(ge=0)
    timestamp: datetime


class ExecutionInput(BaseModel):
    execution_id: str
    order_id: str
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)
    executed_at: datetime


class Scenario(BaseModel):
    scenario_id: str
    version: str = "v0.1"
    security: str = "FINRL"
    description: str | None = None
    order: OrderInput | None = None
    orders: list[OrderInput] = Field(default_factory=list)
    quotes: list[QuoteInput] = Field(default_factory=list)
    executions: list[ExecutionInput] = Field(default_factory=list)

    def get_all_orders(self) -> list[OrderInput]:
        result = []
        if self.order is not None:
            result.append(self.order)
        result.extend(self.orders)
        return result
