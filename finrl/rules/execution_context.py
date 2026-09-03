from datetime import timedelta
from decimal import Decimal

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import OrderSide
from finrl.domain.quote import Quote
from finrl.rules.metrics import share_weighted_realized_spread


def quote_for_execution(
    execution: Execution,
    market: MarketState,
) -> Quote | None:
    return market.quote_at(execution.executed_at)


def quote_at_horizon(
    execution: Execution,
    market: MarketState,
    horizon: timedelta,
) -> Quote | None:
    target_timestamp = execution.executed_at + horizon
    return market.quote_at(target_timestamp)


def realized_spread_at_horizon(
    side: OrderSide,
    executions: list[Execution],
    market: MarketState,
    horizon: timedelta,
) -> Decimal | None:
    future_quotes = {
        execution.execution_id: quote_at_horizon(
            execution,
            market,
            horizon,
        )
        for execution in executions
    }

    return share_weighted_realized_spread(
        side,
        executions,
        future_quotes,
    )


