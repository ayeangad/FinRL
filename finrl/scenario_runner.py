import json
from pathlib import Path

from finrl.domain.execution import Execution
from finrl.domain.market import MarketState
from finrl.domain.order import Order
from finrl.domain.quote import Quote
from finrl.evals.order_evaluator import evaluate_order
from finrl.rules.report_builder import build_rule_605_report
from finrl.rules.rule_605_report import Rule605Report
from finrl.rules.serializer import (
    serialize_rule_605_json,
    serialize_rule_605_pipe_delimited,
)
from finrl.scenario import Scenario


def load_scenario(path_or_data: str | Path | dict) -> Scenario:
    if isinstance(path_or_data, (str, Path)):
        p = Path(path_or_data)
        data = json.loads(p.read_text())
    else:
        data = path_or_data
    return Scenario.model_validate(data)


def parse_scenario(
    scenario: Scenario,
) -> tuple[list[Order], MarketState, list[Execution]]:
    domain_orders = [
        Order(
            order_id=o.order_id,
            security=scenario.security,
            side=o.side,
            order_type=o.order_type,
            quantity=o.quantity,
            limit_price=o.limit_price,
            stop_price=o.stop_price,
            received_at=o.received_at,
        )
        for o in scenario.get_all_orders()
    ]

    domain_quotes = [
        Quote(
            security=q.security,
            bid_price=q.bid_price,
            bid_size=q.bid_size,
            ask_price=q.ask_price,
            ask_size=q.ask_size,
            timestamp=q.timestamp,
        )
        for q in scenario.quotes
    ]

    domain_executions = [
        Execution(
            execution_id=e.execution_id,
            order_id=e.order_id,
            price=e.price,
            quantity=e.quantity,
            executed_at=e.executed_at,
        )
        for e in scenario.executions
    ]

    market = MarketState(security=scenario.security, quotes=domain_quotes)
    return domain_orders, market, domain_executions


def run_scenario(scenario: Scenario) -> Rule605Report:
    orders, market, executions = parse_scenario(scenario)

    order_reports = []
    for order in orders:
        order_executions = [
            ex for ex in executions if ex.order_id == order.order_id
        ]
        report = evaluate_order(order, order_executions, market)
        order_reports.append(report)

    return build_rule_605_report(order_reports)


def run_scenario_and_serialize(
    scenario: Scenario, format: str = "pipe"
) -> str:
    report = run_scenario(scenario)
    if format == "json":
        return serialize_rule_605_json(report)
    return serialize_rule_605_pipe_delimited(report)
