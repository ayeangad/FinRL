from decimal import Decimal
import pytest

from finrl.rules.order_size import OrderSizeBucket, classify_order_size


def test_invalid_quantity_raises_value_error():
    with pytest.raises(ValueError, match="Quantity must be positive"):
        classify_order_size(Decimal("0"))

    with pytest.raises(ValueError, match="Quantity must be positive"):
        classify_order_size(Decimal("-50"))


def test_odd_lot_bucket():
    assert classify_order_size(Decimal("1")) == OrderSizeBucket.ODD_LOT
    assert classify_order_size(Decimal("50")) == OrderSizeBucket.ODD_LOT
    assert classify_order_size(Decimal("99.9")) == OrderSizeBucket.ODD_LOT


def test_shares_100_to_499_bucket():
    assert classify_order_size(Decimal("100")) == OrderSizeBucket.SHARES_100_TO_499
    assert classify_order_size(Decimal("250")) == OrderSizeBucket.SHARES_100_TO_499
    assert classify_order_size(Decimal("499")) == OrderSizeBucket.SHARES_100_TO_499


def test_shares_500_to_1999_bucket():
    assert classify_order_size(Decimal("500")) == OrderSizeBucket.SHARES_500_TO_1999
    assert classify_order_size(Decimal("1000")) == OrderSizeBucket.SHARES_500_TO_1999
    assert classify_order_size(Decimal("1999")) == OrderSizeBucket.SHARES_500_TO_1999


def test_shares_2000_to_4999_bucket():
    assert classify_order_size(Decimal("2000")) == OrderSizeBucket.SHARES_2000_TO_4999
    assert classify_order_size(Decimal("3500")) == OrderSizeBucket.SHARES_2000_TO_4999
    assert classify_order_size(Decimal("4999")) == OrderSizeBucket.SHARES_2000_TO_4999


def test_shares_5000_to_9999_bucket():
    assert classify_order_size(Decimal("5000")) == OrderSizeBucket.SHARES_5000_TO_9999
    assert classify_order_size(Decimal("7500")) == OrderSizeBucket.SHARES_5000_TO_9999
    assert classify_order_size(Decimal("9999")) == OrderSizeBucket.SHARES_5000_TO_9999


def test_shares_10000_plus_bucket():
    assert classify_order_size(Decimal("10000")) == OrderSizeBucket.SHARES_10000_PLUS
    assert classify_order_size(Decimal("50000")) == OrderSizeBucket.SHARES_10000_PLUS
