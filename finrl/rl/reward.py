"""Dense verifiable reward for Rule 605 reports.

Replaces the blunt line-match score with per-column partial credit,
numeric tolerance for float metrics, and a category x bucket confusion
matrix. Output is in [0, 1] and suitable as an RL reward / RLVR signal.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from finrl.rules.serializer import PIPE_DELIMITED_HEADER

REGULATORY_INDICES = (0, 1, 2, 3)
SHARE_VOLUME_INDICES = (4, 5, 6, 7, 8)
HEADER_LEN = len(PIPE_DELIMITED_HEADER)


def _split_rows(pipe: str | None) -> tuple[str, dict[tuple[str, str], list[str]]]:
    if not pipe or not isinstance(pipe, str) or not pipe.strip():
        return "", {}
    lines = [ln.strip() for ln in pipe.strip().split("\n") if ln.strip()]
    if not lines:
        return "", {}
    header = lines[0]
    rows: dict[tuple[str, str], list[str]] = {}
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) >= 2:
            rows[(parts[0], parts[1])] = parts
    return header, rows


def _fields_equal(a: str, b: str, numeric_tolerance: float = 1e-6) -> bool:
    if a == b:
        return True
    try:
        da = Decimal(a)
        db = Decimal(b)
        return abs(float(da) - float(db)) <= numeric_tolerance
    except (InvalidOperation, ValueError, TypeError):
        return False


def dense_report_reward(
    submitted_pipe: str | None,
    ground_truth_pipe: str,
    numeric_tolerance: float = 1e-6,
) -> dict:
    """Compute dense reward + diagnostics. Always returns reward in [0, 1]."""
    gt_header, gt_rows = _split_rows(ground_truth_pipe)
    sub_header, sub_rows = _split_rows(submitted_pipe)

    total_cells = len(gt_rows)
    if total_cells == 0:
        return {
            "reward": 0.0,
            "row_coverage": 0.0,
            "regulatory_accuracy": 0.0,
            "numeric_accuracy": 0.0,
            "per_column_accuracy": {},
            "matched_rows": 0,
            "total_rows": 0,
            "header_match": False,
        }

    if not sub_rows:
        header_only = bool(submitted_pipe and submitted_pipe.strip().split("\n")[0].strip() == gt_header)
        return {
            "reward": 0.05 if header_only else 0.0,
            "row_coverage": 0.0,
            "regulatory_accuracy": 0.0,
            "numeric_accuracy": 0.0,
            "per_column_accuracy": {h: 0.0 for h in PIPE_DELIMITED_HEADER},
            "matched_rows": 0,
            "total_rows": total_cells,
            "header_match": False,
        }

    header_match = sub_header == gt_header
    col_match = [0] * HEADER_LEN
    col_total = [0] * HEADER_LEN
    reg_match = reg_total = 0
    num_match = num_total = 0
    exact_rows = 0

    for key, gt_parts in gt_rows.items():
        sub_parts = sub_rows.get(key)
        if sub_parts is None:
            for i in range(min(len(gt_parts), HEADER_LEN)):
                col_total[i] += 1
            reg_total += len(REGULATORY_INDICES)
            num_total += HEADER_LEN - len(REGULATORY_INDICES)
            continue
        width = min(len(gt_parts), len(sub_parts), HEADER_LEN)
        row_exact = (
            len(gt_parts) == len(sub_parts)
            and all(
                _fields_equal(sub_parts[i], gt_parts[i], numeric_tolerance)
                for i in range(width)
            )
            and width == HEADER_LEN
        )
        if row_exact:
            exact_rows += 1
        for i in range(min(len(gt_parts), HEADER_LEN)):
            col_total[i] += 1
            a = sub_parts[i] if i < len(sub_parts) else ""
            if _fields_equal(a, gt_parts[i], numeric_tolerance):
                col_match[i] += 1
                if i in REGULATORY_INDICES:
                    reg_match += 1
                else:
                    num_match += 1
            else:
                if i in REGULATORY_INDICES:
                    pass
                else:
                    pass
            if i in REGULATORY_INDICES:
                reg_total += 1
            else:
                num_total += 1

    per_column = {
        PIPE_DELIMITED_HEADER[i]: round(col_match[i] / max(1, col_total[i]), 4)
        for i in range(HEADER_LEN)
    }
    reg_acc = round(reg_match / max(1, reg_total), 4)
    num_acc = round(num_match / max(1, num_total), 4)
    row_cov = round(exact_rows / max(1, total_cells), 4)

    # Weighted dense reward: regulatory fields dominate (correctness-critical),
    # numeric fields give smooth partial credit, header + coverage shape the rest.
    reward = round(
        0.45 * reg_acc + 0.35 * num_acc + 0.15 * row_cov + 0.05 * (1.0 if header_match else 0.0),
        4,
    )
    return {
        "reward": float(max(0.0, min(1.0, reward))),
        "row_coverage": row_cov,
        "regulatory_accuracy": reg_acc,
        "numeric_accuracy": num_acc,
        "per_column_accuracy": per_column,
        "matched_rows": exact_rows,
        "total_rows": total_cells,
        "header_match": header_match,
    }


def confusion_matrix(
    submitted_pipe: str | None,
    ground_truth_pipe: str,
) -> dict:
    """Confusion over (category x bucket) row keys.

    Returns {gt_key: {predicted_key or 'MISSING': count}} plus summary counts.
    """
    _, gt_rows = _split_rows(ground_truth_pipe)
    _, sub_rows = _split_rows(submitted_pipe)

    matrix: dict[str, dict[str, int]] = {}
    true_positives = 0
    missing = 0
    spurious = 0
    for key in gt_rows:
        k = f"{key[0]} x {key[1]}"
        if key in sub_rows:
            matrix.setdefault(k, {}).update({"correct": 1})
            true_positives += 1
        else:
            matrix.setdefault(k, {}).update({"MISSING": 1})
            missing += 1
    for key in sub_rows:
        if key not in gt_rows:
            k = f"{key[0]} x {key[1]}"
            matrix.setdefault(k, {}).update({"SPURIOUS": 1})
            spurious += 1
    return {
        "matrix": matrix,
        "true_positives": true_positives,
        "missing": missing,
        "spurious": spurious,
        "total_gt_rows": len(gt_rows),
    }
