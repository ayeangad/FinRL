from datetime import datetime
from decimal import Decimal

from finrl.domain.order import Order, OrderSide, OrderType
from finrl.evals.order_evaluator import evaluate_order
from finrl.scenarios import load_executions, load_market, load_order, validate_scenario

def make_limit_order_scenario(
    *,
    scenario_id: str,
    security: str,
    side: OrderSide,
    quantity: Decimal,
    executions: list[tuple[Decimal, Decimal, datetime]],
    limit_price: Decimal,
    quotes: list[tuple[datetime, Decimal, Decimal, Decimal, Decimal]],
) -> dict:
    received_at = datetime.fromisoformat("2026-09-01T10:30:00.100")

    scenario = {
        "scenario_id" : scenario_id,
        "version" : "v0.1",
        "security" : security,
        "order" : {
            "order_id" : f"{scenario_id}-ORDER",
            "side" : side.value,
            "order_type" : OrderType.LIMIT.value,
            "quantity":str(quantity),
            "limit_price":str(limit_price),
            "received_at": received_at.isoformat(),
        },
        "quotes": make_market_timeline(
            security=security,
            quotes=quotes,
        ),
        "executions": [
            {
                "execution_id": f"{scenario_id}-EXECUTION-{index + 1:03d}",
                "order_id": f"{scenario_id}-ORDER",
                "price": str(execution_price),
                "quantity": str(execution_quantity),
                "executed_at": execution_at.isoformat(),
            }
        for index, (execution_price, execution_quantity, execution_at)
        in enumerate(executions)
    ],
    }


    order = load_order(
        scenario["order"],
        scenario["security"],
    )
    market = load_market(scenario)
    executions = load_executions(scenario["executions"])

    report = evaluate_order(
        order=order,
        executions=executions,
        market=market,
    )

    scenario["expected"] = {
        "executed_quantity": str(report.executed_quantity),
        "average_execution_price": str(report.average_execution_price),
        "quoted_spread": str(report.quoted_spread),
        "price_improvement": str(report.price_improvement),
        "effective_spread": str(report.effective_spread),
    }

    return scenario


def make_market_timeline(
    *,
    security: str,
    quotes: list[tuple[datetime, Decimal, Decimal, Decimal, Decimal]],
) -> list[dict]:
    return [
        {
            "security": security,
            "bid_price": str(bid_price),
            "bid_size": str(bid_size),
            "ask_price": str(ask_price),
            "ask_size": str(ask_size),
            "timestamp": timestamp.isoformat(),
        }
        for timestamp, bid_price, bid_size, ask_price, ask_size in quotes
    ]