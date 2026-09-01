
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator


class OrderExecutionResult(BaseModel):
    order_id: str

    requested_quantity: Decimal = Field(gt=0)
    executed_quantity: Decimal = Field(ge=0)

    average_execution_price: Decimal | None = Field(
        default=None,
        gt=0,
    )

    @model_validator(mode="after")
    def validate_execution_quantity(self):
        if self.executed_quantity > self.requested_quantity:
            raise ValueError(
                "Executed quantity cannot exceed requested quantity"
            )

        return self

    @property
    def fully_executed(self) -> bool:
        return self.executed_quantity == self.requested_quantity

    @property
    def partially_executed(self) -> bool:
        return (
            self.executed_quantity > 0
            and self.executed_quantity < self.requested_quantity
        )

    @property
    def unexecuted(self) -> bool:
        return self.executed_quantity == 0