from datetime import UTC, datetime
from decimal import Decimal

from finrl.benchmark.context_selector import ContextSelector, ScenarioProfile
from finrl.domain.order import OrderSide, OrderType
from finrl.env.state import ObservableOrder


def _obs_order(order_type: OrderType, order_id: str = "O1") -> ObservableOrder:
    return ObservableOrder(
        order_id=order_id,
        security="FINRL",
        side=OrderSide.BUY,
        order_type=order_type,
        quantity=Decimal("100"),
        limit_price=Decimal("10.00") if order_type != OrderType.MARKET else None,
        stop_price=Decimal("9.00") if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) else None,
        received_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
    )


BASE = ("00_role", "01_tools", "02_report_schema", "03_report_formatting", "04_react_protocol")


def test_market_order_selects_market_section(tmp_path):
    selector = ContextSelector(sections_dir=tmp_path)
    for name in (*BASE, "06_market_orders"):
        (tmp_path / f"{name}.txt").write_text(name)
    profile = ScenarioProfile.from_observation("s1", [_obs_order(OrderType.MARKET)])
    assembled = selector.build_system_prompt(profile)
    assert assembled.selected_sections == [*BASE, "06_market_orders"]
    assert "06_market_orders" in assembled.text


def test_stop_order_selects_stop_section(tmp_path):
    selector = ContextSelector(sections_dir=tmp_path)
    for name in (*BASE, "05_stop_orders"):
        (tmp_path / f"{name}.txt").write_text(name)
    profile = ScenarioProfile.from_observation("s2", [_obs_order(OrderType.STOP)])
    assembled = selector.build_system_prompt(profile)
    assert assembled.selected_sections == [*BASE, "05_stop_orders"]


def test_multiple_order_types_selects_multiple_sections(tmp_path):
    selector = ContextSelector(sections_dir=tmp_path)
    for name in (*BASE, "05_stop_orders", "07_limit_orders", "08_multi_order_aggregation"):
        (tmp_path / f"{name}.txt").write_text(name)
    profile = ScenarioProfile.from_observation(
        "s3", [_obs_order(OrderType.STOP_LIMIT), _obs_order(OrderType.LIMIT, "O2")]
    )
    assembled = selector.build_system_prompt(profile)
    assert "05_stop_orders" in assembled.selected_sections
    assert "07_limit_orders" in assembled.selected_sections


def test_multi_order_adds_aggregation_section(tmp_path):
    selector = ContextSelector(sections_dir=tmp_path)
    for name in (*BASE, "06_market_orders", "08_multi_order_aggregation"):
        (tmp_path / f"{name}.txt").write_text(name)
    profile = ScenarioProfile.from_observation(
        "s4", [_obs_order(OrderType.MARKET, "O1"), _obs_order(OrderType.MARKET, "O2")]
    )
    assembled = selector.build_system_prompt(profile)
    assert "08_multi_order_aggregation" in assembled.selected_sections


def test_single_order_omits_aggregation_section(tmp_path):
    selector = ContextSelector(sections_dir=tmp_path)
    for name in (*BASE, "06_market_orders"):
        (tmp_path / f"{name}.txt").write_text(name)
    profile = ScenarioProfile.from_observation("s5", [_obs_order(OrderType.MARKET)])
    assembled = selector.build_system_prompt(profile)
    assert "08_multi_order_aggregation" not in assembled.selected_sections


def test_profile_fields():
    profile = ScenarioProfile.from_observation("s6", [_obs_order(OrderType.LIMIT, "O1")])
    assert profile.num_orders == 1
    assert profile.has_multi_order is False
    assert profile.order_types == {OrderType.LIMIT}
    assert profile.scenario_id == "s6"


def test_missing_section_raises(tmp_path):
    selector = ContextSelector(sections_dir=tmp_path)
    (tmp_path / "00_role.txt").write_text("role")
    profile = ScenarioProfile.from_observation("s7", [_obs_order(OrderType.MARKET)])
    try:
        selector.build_system_prompt(profile)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for missing section")


def test_missing_dir_raises(tmp_path):
    try:
        ContextSelector(sections_dir=tmp_path / "nope")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for missing dir")