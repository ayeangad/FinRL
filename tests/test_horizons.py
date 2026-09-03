from datetime import timedelta

from finrl.rules.horizons import RealizedSpreadHorizon


def test_realized_spread_horizons_have_expected_durations():
    assert RealizedSpreadHorizon.MS_50.duration == timedelta(milliseconds=50)
    assert RealizedSpreadHorizon.S_1.duration == timedelta(seconds=1)
    assert RealizedSpreadHorizon.S_15.duration == timedelta(seconds=15)
    assert RealizedSpreadHorizon.M_1.duration == timedelta(minutes=1)
    assert RealizedSpreadHorizon.M_5.duration == timedelta(minutes=5)


def test_realized_spread_horizon_values_are_distinct():
    values = [horizon.value for horizon in RealizedSpreadHorizon]
    assert len(values) == len(set(values))
    assert len(values) == 5
