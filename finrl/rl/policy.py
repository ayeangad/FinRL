"""Minimal learnable policy: softmax over tools, trained with REINFORCE.

Keeps dependencies to numpy (already in the stack) so training runs on CPU
with no GPU/LLM weights. This is genuine RL: stochastic policy, sampled
rollouts, discounted returns, baseline, gradient step, checkpoint artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from finrl.rl.gym_env import MASKED_TOOLS, TOOLS


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / max(1e-12, e.sum())


class SoftmaxToolPolicy:
    def __init__(self, feature_dim: int = 5, n_actions: int = 3, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.1, size=(feature_dim, n_actions))
        self.feature_dim = feature_dim
        self.n_actions = n_actions

    def probs(self, features: list[float] | np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=float)
        return _softmax(f @ self.W)

    def sample(self, features) -> tuple[int, np.ndarray]:
        p = self.probs(features)
        a = int(self.rng.choice(self.n_actions, p=p))
        return a, p

    def greedy(self, features) -> int:
        return int(np.argmax(self.probs(features)))

    def reinforce_update(
        self,
        trajectories: list[dict],
        lr: float = 0.05,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ) -> dict:
        """One batched REINFORCE step with mean-return baseline.

        trajectories: [{features: [T,F], actions: [T], rewards: [T]}]
        Returns loss/return stats.
        """
        # Baseline = mean episode return (variance reduction, no critic needed).
        returns_all = []
        for t in trajectories:
            r = np.asarray(t["rewards"], dtype=float)
            g = 0.0
            disc = np.zeros_like(r)
            for i in reversed(range(len(r))):
                g = r[i] + gamma * g
                disc[i] = g
            returns_all.append(disc)
        baseline = float(np.mean([d.sum() for d in returns_all])) if returns_all else 0.0

        grad = np.zeros_like(self.W)
        total_loss = 0.0
        total_ret = 0.0
        for t, disc in zip(trajectories, returns_all):
            F = np.asarray(t["features"], dtype=float)  # [T, D]
            A = np.asarray(t["actions"], dtype=int)
            adv = disc - baseline / max(1, len(disc))
            total_ret += float(disc.sum())
            for f, a, ad in zip(F, A, adv):
                p = _softmax(f @ self.W)
                # grad of log pi(a|s) wrt W: outer(f, onehot(a) - p)
                onehot = np.zeros(self.n_actions)
                onehot[a] = 1.0
                grad += np.outer(f, (onehot - p)) * ad
                total_loss += float(-np.log(max(1e-12, p[a])) * ad)
                # entropy bonus gradient: -coef * sum(p log p) derivative approx
                grad += entropy_coef * np.outer(f, (np.ones(self.n_actions) / self.n_actions - p))
        grad /= max(1, sum(len(t["actions"]) for t in trajectories))
        self.W += lr * grad
        n = max(1, len(trajectories))
        return {
            "loss": round(float(total_loss / n), 4),
            "mean_return": round(float(total_ret / n), 4),
            "baseline": round(baseline, 4),
            "episodes": len(trajectories),
        }

    def submit_bias_for_full_coverage(self):
        """Interpretability helper: which action does full coverage prefer?"""
        return self.probs([1.0, 0.03, 1.0, 1.0, 1.0]).tolist()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, tools=np.array(TOOLS))
        return path

    @classmethod
    def load(cls, path: str | Path) -> SoftmaxToolPolicy:
        data = np.load(path, allow_pickle=True)
        W = data["W"]
        pol = cls(feature_dim=W.shape[0], n_actions=W.shape[1])
        pol.W = W
        return pol


SUBMIT_HINT = (
    "Policy learns: gather classify+metrics per order first "
    f"(masked actions 0,1 = {MASKED_TOOLS[0]},{MASKED_TOOLS[1]}), "
    f"then submit (masked action 2). Full 7-tool space in TOOLS."
)
