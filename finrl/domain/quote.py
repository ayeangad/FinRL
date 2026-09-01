from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

class Quote(BaseModel):
    security: str
    
    bid_price: Decimal = Field(gt=0)
    bid_size: int = Field(gt=0)
    
    ask_price: Decimal = Field(gt=0)
    ask_size: int = Field(gt=0)
    
    timestamp: datetime
    
    