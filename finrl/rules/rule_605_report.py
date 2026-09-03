from pydantic import BaseModel, Field

from finrl.rules.category_report import CategoryReport
from finrl.rules.classification import OrderTypeCategory
from finrl.rules.order_size import OrderSizeBucket


class Rule605Report(BaseModel):
    categories: dict[tuple[OrderTypeCategory, OrderSizeBucket], CategoryReport] = Field(
        default_factory=dict
    )

    def get_cell(
        self,
        category: OrderTypeCategory,
        size_bucket: OrderSizeBucket,
    ) -> CategoryReport:
        return self.categories[(category, size_bucket)]

    def all_cells(self) -> list[CategoryReport]:
        return list(self.categories.values())
