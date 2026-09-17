"""Seeded procedural Rule 605 scenario generator.

Produces valid Scenario JSON (pydantic-validated) with controllable
diversity: order types, sides, size buckets, limit regimes, multi-order
counts. Every generated order is executable/reportable by construction so
the dense reward is learnable (no dead scenarios).

Usage:
  python -m finrl.scenario_gen --n 50 --seed 0 --out-dir scenarios/v0.1/synth --prefix synth
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from finrl.domain.order import OrderSide, OrderType

BASE_TS = datetime.fromisoformat("2026-09-01T10:30:00")
BASE_BID = Decimal("99.90")
BASE_ASK = Decimal("100.10")

BUCKET_RANGES = [
    (1, 99),
    (100, 499),
    (500, 1999),
    (2000, 4999),
    (5000, 9999),
    (10000, 20000),
]


def _jitter_price(rng, base: Decimal, ticks: int = 4) -> Decimal:
    # +/- ticks cents.
    delta_cents = int(rng.integers(-ticks, ticks + 1))
    return (base + Decimal(delta_cents) / Decimal(100)).quantize(Decimal("0.01"))


def _sample_quantity(rng) -> int:
    lo, hi = BUCKET_RANGES[int(rng.integers(0, len(BUCKET_RANGES)))]
    return int(rng.integers(lo, hi + 1))


def _limit_for_regime(rng, side: OrderSide, bid: Decimal, ask: Decimal) -> Decimal:
    mid = (bid + ask) / 2
    regime = int(rng.integers(0, 3))  # 0 marketable, 1 midpoint, 2 non-marketable
    if side == OrderSide.BUY:
        if regime == 0:
            return ask + Decimal(int(rng.integers(0, 5))) / Decimal(100)
        if regime == 1:
            return mid
        # non-marketable but still >= bid (executable by construction)
        return bid + (mid - bid) / 2
    else:
        if regime == 0:
            return bid - Decimal(int(rng.integers(0, 5))) / Decimal(100)
        if regime == 1:
            return mid
        return ask - (ask - mid) / 2


def generate_scenario(
    scenario_id: str,
    n_orders: int = 1,
    seed: int = 0,
    security: str = "FINRL",
) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    orders: list[dict] = []
    quotes: list[dict] = []
    executions: list[dict] = []
    for i in range(n_orders):
        ts = BASE_TS + timedelta(seconds=i * 5)
        side = OrderSide.BUY if rng.random() < 0.5 else OrderSide.SELL
        otype = OrderType(str(rng.choice(["market", "limit", "stop", "stop_limit"])))
        qty = _sample_quantity(rng)
        bid = _jitter_price(rng, BASE_BID)
        ask = _jitter_price(rng, BASE_ASK)
        if ask <= bid:
            ask = bid + Decimal("0.20")
        oid = f"{scenario_id}-O{i + 1}"
        limit_price = None
        stop_price = None
        if otype == OrderType.LIMIT:
            limit_price = _limit_for_regime(rng, side, bid, ask).quantize(Decimal("0.01"))
        elif otype == OrderType.STOP:
            mid = (bid + ask) / 2
            stop_price = mid.quantize(Decimal("0.01"))
        elif otype == OrderType.STOP_LIMIT:
            mid = (bid + ask) / 2
            stop_price = mid.quantize(Decimal("0.01"))
            # marketable limit after trigger (executable by construction)
            if side == OrderSide.BUY:
                limit_price = (ask + Decimal("0.05")).quantize(Decimal("0.01"))
            else:
                limit_price = (bid - Decimal("0.05")).quantize(Decimal("0.01"))
        # Execution at touch (full fill) -> reportable.
        fill = ask if side == OrderSide.BUY else bid
        orders.append(
            {
                "order_id": oid,
                "side": side.value,
                "order_type": otype.value,
                "quantity": str(qty),
                "limit_price": str(limit_price) if limit_price is not None else None,
                "stop_price": str(stop_price) if stop_price is not None else None,
                "received_at": ts.isoformat(),
            }
        )
        quotes.append(
            {
                "security": security,
                "bid_price": str(bid),
                "bid_size": "500",
                "ask_price": str(ask),
                "ask_size": "300",
                "timestamp": ts.isoformat(),
            }
        )
        executions.append(
            {
                "execution_id": f"{scenario_id}-E{i + 1}",
                "order_id": oid,
                "price": str(fill),
                "quantity": str(qty),
                "executed_at": ts.isoformat(),
            }
        )
    data = {
        "scenario_id": scenario_id,
        "version": "v0.1",
        "security": security,
        "description": f"Synthetic seed={seed} n_orders={n_orders}",
        "orders": orders,
        "quotes": quotes,
        "executions": executions,
    }
    # Validate eagerly so bad generations fail fast.
    from finrl.scenario import Scenario

    Scenario.model_validate(data)
    return data


def generate_batch(
    out_dir: str | Path = "scenarios/v0.1/synth",
    n: int = 50,
    seed: int = 0,
    prefix: str = "synth",
    min_orders: int = 1,
    max_orders: int = 4,
) -> list[Path]:
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for i in range(n):
        n_orders = int(rng.integers(min_orders, max_orders + 1))
        sid = f"{prefix}_{i:03d}"
        data = generate_scenario(sid, n_orders=n_orders, seed=int(seed * 1000 + i))
        p = out / f"{sid}.json"
        p.write_text(json.dumps(data, indent=2))
        paths.append(p)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Rule 605 scenarios")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="scenarios/v0.1/synth")
    p.add_argument("--prefix", default="synth")
    p.add_argument("--min-orders", type=int, default=1)
    p.add_argument("--max-orders", type=int, default=4)
    args = p.parse_args()
    paths = generate_batch(args.out_dir, args.n, args.seed, args.prefix, args.min_orders, args.max_orders)
    print(f"Wrote {len(paths)} scenarios to {args.out_dir}")


if __name__ == "__main__":
    main()
