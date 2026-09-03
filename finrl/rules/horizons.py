from datetime import timedelta
from enum import Enum


class RealizedSpreadHorizon(str, Enum):
    MS_50 = "50ms"
    S_1 = "1s"
    S_15 = "15s"
    M_1 = "1m"
    M_5 = "5m"

    @property
    def duration(self) -> timedelta:
        durations = {
            RealizedSpreadHorizon.MS_50: timedelta(milliseconds=50),
            RealizedSpreadHorizon.S_1: timedelta(seconds=1),
            RealizedSpreadHorizon.S_15: timedelta(seconds=15),
            RealizedSpreadHorizon.M_1: timedelta(minutes=1),
            RealizedSpreadHorizon.M_5: timedelta(minutes=5),
        }
        return durations[self]
