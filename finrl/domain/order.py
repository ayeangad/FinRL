from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    
class Order(BaseModel):
    order_id: str
    security: str

    side: OrderSide
    order_type: OrderType
    
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = Field(gt=0, default=None)
    stop_price: Decimal | None = Field(gt=0, default=None)

    received_at: datetime


    


