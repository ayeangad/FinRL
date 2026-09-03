from decimal import Decimal

from pydantic import BaseModel, Field

from finrl.rules.classification import OrderTypeCategory
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.order_size import OrderSizeBucket


class OrderReport(BaseModel):
    order_id: str
    order_size_bucket: OrderSizeBucket
    order_type_category: OrderTypeCategory
    reportable: bool

    requested_quantity: Decimal = Field(gt=0)
    executed_quantity: Decimal = Field(ge=0)

    average_execution_price: Decimal | None = Field(
        default=None,
        gt=0,
    )

    price_improvement: Decimal | None = None
    effective_spread: Decimal | None = None
    quoted_spread: Decimal | None = None

    realized_spreads: dict[RealizedSpreadHorizon, Decimal | None] = Field(
        default_factory=dict
    )

    percentage_effective_spread: Decimal | None = None
    percentage_quoted_spread: Decimal | None = None

    percentage_realized_spreads: dict[RealizedSpreadHorizon, Decimal | None] = Field(
        default_factory=dict
    )