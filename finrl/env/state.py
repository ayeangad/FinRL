from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from finrl.domain.order import OrderSide, OrderType


class ObservableOrder(BaseModel):
    order_id: str
    security: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    received_at: datetime


class ObservableQuote(BaseModel):
    security: str
    bid_price: Decimal
    bid_size: Decimal
    ask_price: Decimal
    ask_size: Decimal
    timestamp: datetime


class ObservableExecution(BaseModel):
    execution_id: str
    order_id: str
    price: Decimal
    quantity: Decimal
    executed_at: datetime


class EnvObservation(BaseModel):
    scenario_id: str
    security: str
    orders: list[ObservableOrder]
    quotes_count: int
    executions_count: int
    current_step: int
    action_history: list[str] = Field(default_factory=list)
