from decimal import Decimal

from pydantic import BaseModel, Field


class OrderReport(BaseModel):
    order_id: str

    requested_quantity: Decimal = Field(gt=0)
    executed_quantity: Decimal = Field(ge=0)

    average_execution_price: Decimal | None = Field(
        default=None,
        gt=0,
    )

    price_improvement: Decimal | None = None
    effective_spread: Decimal | None = None
    quoted_spread: Decimal | None = None