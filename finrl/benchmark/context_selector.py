from dataclasses import dataclass, field
from pathlib import Path

from finrl.domain.order import OrderType
from finrl.env.state import ObservableOrder


@dataclass
class ScenarioProfile:
    scenario_id: str
    num_orders: int = 0
    has_multi_order: bool = False
    order_types: set[OrderType] = field(default_factory=set)

    @classmethod
    def from_observation(cls, scenario_id: str, orders: list[ObservableOrder] | list[dict]) -> "ScenarioProfile":
        order_types = set()
        for o in orders:
            raw_type = o.order_type if isinstance(o, ObservableOrder) else o.get("order_type")
            try:
                order_types.add(OrderType(raw_type))
            except (ValueError, TypeError):
                continue
        return cls(
            scenario_id=scenario_id,
            num_orders=len(orders),
            has_multi_order=len(orders) > 1,
            order_types=order_types,
        )


@dataclass
class AssembledPrompt:
    text: str
    selected_sections: list[str] = field(default_factory=list)


class ContextSelector:
    """Scenario-aware SEC Rule 605 prompt assembly.

    Always includes the task, tool, schema, formatting, and ReAct sections.
    Additionally injects only the rule sections relevant to the order types
    present in a scenario (plus multi-order aggregation rules when needed).
    """

    BASE_SECTIONS: tuple[str, ...] = (
        "00_role",
        "01_tools",
        "02_report_schema",
        "03_report_formatting",
        "04_react_protocol",
    )

    CONDITIONAL_BY_ORDER_TYPE: dict[OrderType, str] = {
        OrderType.MARKET: "06_market_orders",
        OrderType.LIMIT: "07_limit_orders",
        OrderType.STOP: "05_stop_orders",
        OrderType.STOP_LIMIT: "05_stop_orders",
    }

    MULTI_ORDER_SECTION = "08_multi_order_aggregation"

    def __init__(self, sections_dir: Path | str = "prompts/sections"):
        self.sections_dir = Path(sections_dir)
        self._cache: dict[str, str] = {}
        if not self.sections_dir.is_dir():
            raise FileNotFoundError(f"Sections dir not found: {self.sections_dir}")

    def _load(self, section_name: str) -> str:
        if section_name not in self._cache:
            path = self.sections_dir / f"{section_name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Prompt section not found: {path}")
            self._cache[section_name] = path.read_text().rstrip()
        return self._cache[section_name]

    def build_system_prompt(self, profile: ScenarioProfile) -> AssembledPrompt:
        selected = list(self.BASE_SECTIONS)

        for order_type in profile.order_types:
            section = self.CONDITIONAL_BY_ORDER_TYPE.get(order_type)
            if section and section not in selected:
                selected.append(section)

        if profile.has_multi_order:
            selected.append(self.MULTI_ORDER_SECTION)

        parts = [self._load(name) for name in selected]
        return AssembledPrompt(text="\n\n".join(parts), selected_sections=selected)

    def build_from_observation(
        self, scenario_id: str, orders: list[ObservableOrder]
    ) -> AssembledPrompt:
        return self.build_system_prompt(ScenarioProfile.from_observation(scenario_id, orders))