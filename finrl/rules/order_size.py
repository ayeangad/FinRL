from decimal import Decimal
from enum import Enum


class OrderSizeBucket(str, Enum):
    ODD_LOT = "1_to_99"
    SHARES_100_TO_499 = "100_to_499"
    SHARES_500_TO_1999 = "500_to_1999"
    SHARES_2000_TO_4999 = "2000_to_4999"
    SHARES_5000_TO_9999 = "5000_to_9999"
    SHARES_10000_PLUS = "10000_plus"


def classify_order_size(quantity: Decimal) -> OrderSizeBucket:
    if quantity <= Decimal("0"):
        raise ValueError("Quantity must be positive")

    if quantity < Decimal("100"):
        return OrderSizeBucket.ODD_LOT
    if quantity < Decimal("500"):
        return OrderSizeBucket.SHARES_100_TO_499
    if quantity < Decimal("2000"):
        return OrderSizeBucket.SHARES_500_TO_1999
    if quantity < Decimal("5000"):
        return OrderSizeBucket.SHARES_2000_TO_4999
    if quantity < Decimal("10000"):
        return OrderSizeBucket.SHARES_5000_TO_9999

    return OrderSizeBucket.SHARES_10000_PLUS
