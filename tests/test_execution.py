from datetime import datetime
from decimal import Decimal

from finrl.domain.execution import Execution


def test_create_execution():
    execution = Execution(
        execution_id="EXE-0001",
        order_id="ORD-0001",
        price=Decimal("100.00"),
        quantity=Decimal("50"),
        executed_at=datetime.fromisoformat(
            "2026-09-01T10:30:00.250"
        ),
    )

    assert execution.execution_id == "EXE-0001"
    assert execution.order_id == "ORD-0001"
    assert execution.price == Decimal("100.00")
    assert execution.quantity == Decimal("50")
