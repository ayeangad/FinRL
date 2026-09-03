from typing import Any
from pydantic import BaseModel, Field

from finrl.rules.serializer import PIPE_DELIMITED_HEADER


class EvaluationDetail(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    success: bool
    critical_errors: int = 0
    major_errors: int = 0
    minor_errors: int = 0
    numeric_accuracy: float = Field(ge=0.0, le=1.0)
    regulatory_accuracy: float = Field(ge=0.0, le=1.0)
    evidence_accuracy: float = Field(ge=0.0, le=1.0)
    workflow_integrity: float = Field(ge=0.0, le=1.0)
    error_summary: list[str] = Field(default_factory=list)


def evaluate_submission(
    submitted_pipe: str | None,
    ground_truth_pipe: str,
) -> EvaluationDetail:
    if not submitted_pipe or not isinstance(submitted_pipe, str):
        return EvaluationDetail(
            score=0.0,
            success=False,
            critical_errors=1,
            numeric_accuracy=0.0,
            regulatory_accuracy=0.0,
            evidence_accuracy=0.0,
            workflow_integrity=0.0,
            error_summary=["No submission or non-string submission."],
        )

    sub_lines = [line.strip() for line in submitted_pipe.strip().split("\n") if line.strip()]
    gt_lines = [line.strip() for line in ground_truth_pipe.strip().split("\n") if line.strip()]

    if submitted_pipe.strip() == ground_truth_pipe.strip():
        return EvaluationDetail(
            score=1.0,
            success=True,
            critical_errors=0,
            major_errors=0,
            minor_errors=0,
            numeric_accuracy=1.0,
            regulatory_accuracy=1.0,
            evidence_accuracy=1.0,
            workflow_integrity=1.0,
            error_summary=[],
        )

    errors = []
    critical = 0
    major = 0
    minor = 0

    total_reg_fields = 0
    matching_reg_fields = 0
    total_num_fields = 0
    matching_num_fields = 0

    # Header check
    if len(sub_lines) == 0 or sub_lines[0] != gt_lines[0]:
        critical += 1
        errors.append("Pipe header missing or incorrect.")

    # Match rows by (category, bucket) key
    gt_rows = {}
    for line in gt_lines[1:]:
        parts = line.split("|")
        if len(parts) >= 2:
            gt_rows[(parts[0], parts[1])] = parts

    sub_rows = {}
    for line in sub_lines[1:]:
        parts = line.split("|")
        if len(parts) >= 2:
            sub_rows[(parts[0], parts[1])] = parts

    for key, gt_parts in gt_rows.items():
        if key not in sub_rows:
            critical += 1
            errors.append(f"Missing category cell row for {key}.")
            continue

        sub_parts = sub_rows[key]
        if len(sub_parts) != len(gt_parts):
            major += 1
            errors.append(f"Cell {key} has incorrect field count ({len(sub_parts)} vs {len(gt_parts)}).")
            continue

        # Field-by-field evaluation
        # Regulatory fields (Category, Bucket, Covered Orders, Executed Orders) -> indices 0, 1, 2, 3
        for idx in [0, 1, 2, 3]:
            total_reg_fields += 1
            if sub_parts[idx] == gt_parts[idx]:
                matching_reg_fields += 1
            else:
                critical += 1
                errors.append(f"Cell {key} regulatory field {PIPE_DELIMITED_HEADER[idx]} mismatch: '{sub_parts[idx]}' vs '{gt_parts[idx]}'.")

        # Share volume fields -> indices 4, 5, 6, 7, 8
        for idx in [4, 5, 6, 7, 8]:
            total_num_fields += 1
            if sub_parts[idx] == gt_parts[idx]:
                matching_num_fields += 1
            else:
                major += 1
                errors.append(f"Cell {key} share volume field {PIPE_DELIMITED_HEADER[idx]} mismatch: '{sub_parts[idx]}' vs '{gt_parts[idx]}'.")

        # Spread & percentage metrics -> indices 9..23
        for idx in range(9, min(len(sub_parts), len(gt_parts))):
            total_num_fields += 1
            if sub_parts[idx] == gt_parts[idx]:
                matching_num_fields += 1
            else:
                minor += 1
                errors.append(f"Cell {key} metric field {PIPE_DELIMITED_HEADER[idx]} mismatch: '{sub_parts[idx]}' vs '{gt_parts[idx]}'.")

    reg_acc = round(matching_reg_fields / max(1, total_reg_fields), 4)
    num_acc = round(matching_num_fields / max(1, total_num_fields), 4)

    penalty = critical * 0.15 + major * 0.05 + minor * 0.01
    score = max(0.0, round(1.0 - penalty, 4))
    success = score >= 0.95 and critical == 0

    return EvaluationDetail(
        score=score,
        success=success,
        critical_errors=critical,
        major_errors=major,
        minor_errors=minor,
        numeric_accuracy=num_acc,
        regulatory_accuracy=reg_acc,
        evidence_accuracy=reg_acc,
        workflow_integrity=1.0 if len(sub_lines) == len(gt_lines) else 0.5,
        error_summary=errors,
    )
