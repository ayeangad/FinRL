from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order
from finrl.rules.aggregation import (
    total_executed_quantity,
    volume_weighted_average_execution_price,
)
from finrl.rules.classification import OrderTypeCategory, classify_order_category
from finrl.rules.eligibility import is_reportable_order
from finrl.rules.execution_context import (
    percentage_realized_spread_at_horizon,
    realized_spread_at_horizon,
)
from finrl.rules.horizons import RealizedSpreadHorizon
from finrl.rules.metrics import (
    effective_spread,
    percentage_quoted_spread,
    price_improvement,
    quoted_spread,
    share_weighted_effective_spread,
    share_weighted_percentage_effective_spread,
    share_weighted_price_improvement,
)
from finrl.rules.order_result import OrderExecutionResult
from finrl.rules.order_size import classify_order_size
from finrl.rules.price_improvement import categorize_execution_shares
from finrl.rules.report import OrderReport


def evaluate_order(
    order: Order,
    executions: list[Execution],
    market: MarketState,
) -> OrderReport:
    executed_quantity = total_executed_quantity(executions)
    average_price = volume_weighted_average_execution_price(executions)
    size_bucket = classify_order_size(order.quantity)
    reportable = is_reportable_order(order, executions, market)
    quote = market.quote_at(order.received_at)

    try:
        category = classify_order_category(order, quote)
    except ValueError:
        category = OrderTypeCategory.NON_MARKETABLE_LIMIT

    if not reportable:
        return OrderReport(
            order_id=order.order_id,
            order_size_bucket=size_bucket,
            order_type_category=category,
            reportable=False,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            average_execution_price=average_price,
            price_improvement=None,
            effective_spread=None,
            quoted_spread=None,
            realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
            percentage_effective_spread=None,
            percentage_quoted_spread=None,
            percentage_realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
        )

    if average_price is None:
        return OrderReport(
            order_id=order.order_id,
            order_size_bucket=size_bucket,
            order_type_category=category,
            reportable=True,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            average_execution_price=None,
            price_improvement=None,
            effective_spread=None,
            quoted_spread=None,
            realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
            percentage_effective_spread=None,
            percentage_quoted_spread=None,
            percentage_realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
        )

    if quote is None:
        return OrderReport(
            order_id=order.order_id,
            order_size_bucket=size_bucket,
            order_type_category=category,
            reportable=False,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            average_execution_price=average_price,
            price_improvement=None,
            effective_spread=None,
            quoted_spread=None,
            realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
            percentage_effective_spread=None,
            percentage_quoted_spread=None,
            percentage_realized_spreads={
                horizon: None for horizon in RealizedSpreadHorizon
            },
        )

    execution_result = OrderExecutionResult(
        order_id=order.order_id,
        requested_quantity=order.quantity,
        executed_quantity=executed_quantity,
        average_execution_price=average_price,
    )

    realized_spreads = {
        horizon: realized_spread_at_horizon(
            order.side,
            executions,
            market,
            horizon.duration,
        )
        for horizon in RealizedSpreadHorizon
    }

    percentage_realized_spreads = {
        horizon: percentage_realized_spread_at_horizon(
            order.side,
            executions,
            market,
            horizon.duration,
            quote,
        )
        for horizon in RealizedSpreadHorizon
    }

    improved_shares, at_quote_shares, outside_shares = categorize_execution_shares(
        order.side,
        executions,
        quote,
    )

    return OrderReport(
        order_id=order.order_id,
        order_size_bucket=size_bucket,
        order_type_category=category,
        reportable=True,
        requested_quantity=order.quantity,
        executed_quantity=execution_result.executed_quantity,
        average_execution_price=execution_result.average_execution_price,
        price_improvement=share_weighted_price_improvement(
            order.side,
            executions,
            quote,
        ),
        effective_spread=share_weighted_effective_spread(
            order.side,
            executions,
            quote,
        ),
        quoted_spread=quoted_spread(quote),
        realized_spreads=realized_spreads,
        percentage_effective_spread=share_weighted_percentage_effective_spread(
            order.side,
            executions,
            quote,
        ),
        percentage_quoted_spread=percentage_quoted_spread(quote),
        percentage_realized_spreads=percentage_realized_spreads,
        shares_price_improved=improved_shares,
        shares_at_quote=at_quote_shares,
        shares_outside_quote=outside_shares,
    )
