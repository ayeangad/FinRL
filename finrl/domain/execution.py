from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class Execution(BaseModel):
    order_id: str
    execution_id: str
    
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0)
    
    executed_at: datetime
