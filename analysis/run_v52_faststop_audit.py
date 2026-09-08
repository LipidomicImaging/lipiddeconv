#!/usr/bin/env python3
"""Descriptively audit convergence in completed v50 K=1/K=3/K=5 runs.

No solver is run and no stopping rule is selected.  Input discovery is
strictly limited to the v50 result root and its immediate case directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = ROOT / "results/v50_reoptimized_sanity_k1_k3"
OUTPUT_DEFAULT = ROOT / "results/v52_faststop_audit"
REQUIRED_CASES = ("k1", "k3", "k5")
FINAL_BANDS = (0.10, 0.05, 0.02, 0.01)
PROGRESS_TARGETS = (500, 800, 1000, 1250, 1500, 2000, 2500, 3000)
COMPOSITION_TARGETS = (1000, 1500, 2000, 3000)
X_DISTANCE_BANDS = (0.05, 0.02, 0.01, 0.005)
SUPPORT_GATES = (1.0e-4, 1.0e-3)
EXPECTED_CANDIDATES = 391


def json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"INVALID_HISTORY: expected JSON object at {path}")
    return payload


def case_token(text: str) -> str | None:
    lowered = text.lower()
    for case in REQUIRED_CASES:
        if re.search(rf"(?:^|[^a-z0-9]){case}(?!\d)", lowered):
            return case
    return None


def report_case_token(report: dict) -> str | None:
    for key in ("case_name", "case", "name"):
        if key in report:
            token = case_token(str(report[key]))
            if token:
                return token
    active = report.get("active_indices")
    if isinstance(active, list) and len(active) in (1, 3, 5):
        return f"k{len(active)}"
    return None


def discover_inputs(input_root: Path) -> dict[str, dict]:
    """Inspect only input_root and one immediate directory level."""
    inventory = {
        case: {
            "case": case,
            "status": "MISSING",
            "history_path": None,
            "report_path": None,
        }
        for case in REQUIRED_CASES
    }
    if not input_root.exists() or not input_root.is_dir():
        return inventory

    candidates = []
    root_files = [path for path in input_root.iterdir() if path.is_file()]
    for path in root_files:
        name = path.name.lower()
        if "training_history" in name:
            token = case_token(name)
            if token:
                report_matches = [
                    other for other in root_files
                    if "report" in other.name.lower()
                    and case_token(other.name) == token
                ]
                candidates.append((token, path, report_matches[0] if report_matches else None))

    for directory in sorted(
        (path for path in input_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        files = [path for path in directory.iterdir() if path.is_file()]
        histories = [path for path in files if "training_history" in path.name.lower()]
        reports = [path for path in files if path.name.lower() == "report.json"]
        reports += [
            path for path in files
            if "report" in path.name.lower() and path not in reports
        ]
        for history_path in histories:
            report_path = reports[0] if reports else None
            token = case_token(directory.name) or case_token(history_path.name)
            if report_path is not None:
                try:
                    token = report_case_token(read_json(report_path)) or token
                except (OSError, ValueError, RuntimeError):
                    pass
            if token:
                candidates.append((token, history_path, report_path))

    for token, history_path, report_path in candidates:
        current = inventory[token]
        if current["history_path"] is None:
            current.update({
                "status": "FOUND" if report_path is not None else "HISTORY_FOUND_REPORT_MISSING",
                "history_path": str(history_path.resolve()),
                "report_path": str(report_path.resolve()) if report_path else None,
            })
    return inventory


def value_schema(value: Any) -> dict:
    if isinstance(value, list):
        result = {"type": "list", "length": len(value)}
        if not value:
            result["element_type"] = "unavailable_empty"
            return result
        non_null = [item for item in value if item is not None]
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in non_null):
            result["element_type"] = "scalar_numeric"
        elif all(isinstance(item, list) for item in non_null):
            widths = sorted({len(item) for item in non_null})
            result.update({"element_type": "list", "element_widths": widths})
        elif all(isinstance(item, dict) for item in non_null):
            result.update({
                "element_type": "object",
                "element_fields": sorted({key for item in non_null for key in item}),
            })
        else:
            result["element_type"] = sorted({type(item).__name__ for item in non_null})
        return result
    if isinstance(value, dict):
        return {"type": "object", "fields": sorted(value)}
    return {"type": type(value).__name__}


def numeric_scalar_list(value: Any, length: int) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    converted = []
    for item in value:
        if item is None:
            converted.append(np.nan)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            converted.append(float(item))
        else:
            return None
    return np.asarray(converted, dtype=np.float64)


def component_labels(key: str, width: int, report: dict) -> list[str]:
    if "raw_losses" in key.lower():
        final_raw = report.get("final_raw_losses")
        if isinstance(final_raw, dict) and len(final_raw) == width:
            return [str(name) for name in final_raw]
    return [f"component_{index}" for index in range(width)]


def extract_scalar_series(
    history: dict, report: dict, epochs: np.ndarray
) -> dict[str, np.ndarray]:
    length = len(epochs)
    series = {}
    for key, value in history.items():
        if key == "epoch":
            continue
        scalar = numeric_scalar_list(value, length)
        if scalar is not None:
            series[key] = scalar
            continue
        if not isinstance(value, list) or len(value) != length or not value:
            continue
        non_null = [item for item in value if item is not None]
        if not non_null or not all(isinstance(item, dict) for item in non_null):
            pass
        else:
            fields = sorted({field for item in non_null for field in item})
            for field in fields:
                component = numeric_scalar_list(
                    [item.get(field) if isinstance(item, dict) else None for item in value],
                    length,
                )
                if component is not None:
                    series[f"{key}.{field}"] = component
            continue
        if not all(isinstance(item, list) for item in non_null):
            continue
        widths = {len(item) for item in non_null}
        if len(widths) != 1:
            continue
        width = widths.pop()
        if width > 16:
            continue
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if matrix.shape != (length, width):
            continue
        for index, label in enumerate(component_labels(key, width, report)):
            series[f"{key}.{label}"] = matrix[:, index]
    return series


def history_epochs(history: dict) -> np.ndarray | None:
    value = history.get("epoch")
    if not isinstance(value, list) or not value:
        return None
    try:
        epochs = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if epochs.ndim != 1 or not np.all(np.isfinite(epochs)) or np.any(np.diff(epochs) <= 0):
        return None
    return epochs


def cadence_summary(epochs: np.ndarray | None) -> dict:
    if epochs is None or len(epochs) < 2:
        return {"status": "UNAVAILABLE"}
    differences = np.diff(epochs)
    return {
        "status": "AVAILABLE",
        "first_recorded_epoch": json_number(epochs[0]),
        "last_recorded_epoch": json_number(epochs[-1]),
        "checkpoint_count": int(len(epochs)),
        "median_epoch_spacing": json_number(np.median(differences)),
        "minimum_epoch_spacing": json_number(differences.min()),
        "maximum_epoch_spacing": json_number(differences.max()),
        "unique_epoch_spacings": [json_number(value) for value in np.unique(differences)],
    }


def metric_roles(metric_names: list[str]) -> dict[str, list[str]]:
    roles = {
        "total_loss": [],
        "physical_loss": [],
        "reconstruction_loss": [],
        "tv_loss": [],
        "sparsity_loss": [],
        "x_relative_change": [],
        "learning_rate": [],
    }
    for name in metric_names:
        lowered = name.lower()
        if "physical" in lowered:
            roles["physical_loss"].append(name)
        if "reconstruction" in lowered or lowered.endswith(".reconstruction"):
            roles["reconstruction_loss"].append(name)
        if re.search(r"(?:^|[._])tv(?:$|[._])", lowered):
            roles["tv_loss"].append(name)
        if "spars" in lowered or "l1" in lowered:
            roles["sparsity_loss"].append(name)
        if "relative_abundance_change" in lowered or "x_relative" in lowered:
            roles["x_relative_change"].append(name)
        if lowered in ("lr", "learning_rate") or "learning_rate" in lowered:
            roles["learning_rate"].append(name)
        if "total" in lowered and "physical" not in lowered:
            roles["total_loss"].append(name)
    return roles


def earliest_final_band(
    epochs: np.ndarray, values: np.ndarray, fraction: float
) -> dict:
    finite = np.isfinite(values)
    if not finite.any():
        return {"status": "UNAVAILABLE"}
    last = np.flatnonzero(finite)[-1]
    final = float(values[last])
    finite_values = np.abs(values[finite])
    scale = max(float(finite_values.max()), np.finfo(np.float64).tiny)
    near_zero = abs(final) <= np.finfo(np.float64).eps * 100.0 * scale
    tolerance = fraction * abs(final)
    deviations = np.abs(values - final)
    eligible = []
    for index in np.flatnonzero(finite):
        tail = deviations[index : last + 1]
        tail_finite = np.isfinite(tail)
        if tail_finite.any() and np.all(tail[tail_finite] <= tolerance):
            eligible.append(index)
    if not eligible:
        return {
            "status": "UNAVAILABLE",
            "final_value": final,
            "relative_to_final_numerically_unstable_near_zero": near_zero,
            "absolute_tolerance": tolerance,
        }
    index = eligible[0]
    return {
        "status": "AVAILABLE",
        "earliest_epoch": json_number(epochs[index]),
        "value_at_earliest_epoch": json_number(values[index]),
        "final_epoch": json_number(epochs[last]),
        "final_value": final,
        "absolute_deviation_at_earliest_epoch": json_number(deviations[index]),
        "absolute_tolerance": tolerance,
        "relative_to_final_numerically_unstable_near_zero": near_zero,
    }


def nearest_checkpoint(
    epochs: np.ndarray, values: np.ndarray, target: float
) -> tuple[int, float] | None:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return None
    index = min(finite.tolist(), key=lambda idx: (abs(epochs[idx] - target), epochs[idx]))
    return index, float(values[index])


def progress_table(
    epochs: np.ndarray, values: np.ndarray, metric: str
) -> list[dict]:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return []
    initial_index, final_index = int(finite[0]), int(finite[-1])
    initial, final = float(values[initial_index]), float(values[final_index])
    improvement = initial - final
    targets = [*PROGRESS_TARGETS, int(epochs[final_index])]
    rows = []
    for target in dict.fromkeys(targets):
        nearest = nearest_checkpoint(epochs, values, target)
        if nearest is None:
            continue
        index, value = nearest
        fraction = None
        if abs(improvement) > np.finfo(np.float64).eps * max(abs(initial), abs(final), 1.0):
            fraction = (initial - value) / improvement
        rows.append({
            "metric": metric,
            "target_epoch": int(target),
            "recorded_epoch": json_number(epochs[index]),
            "value": value,
            "initial_value": initial,
            "final_value": final,
            "fraction_of_total_final_improvement": json_number(fraction) if fraction is not None else None,
        })
    return rows


def preferred_metric(roles: dict[str, list[str]], role: str) -> str | None:
    names = roles.get(role, [])
    if not names:
        return None
    preferences = {
        "total_loss": ("eval_total", "train_total"),
        "physical_loss": ("eval_physical_loss",),
        "reconstruction_loss": ("eval_raw_losses.reconstruction",),
        "tv_loss": ("eval_raw_losses.tv",),
    }
    for preferred in preferences.get(role, ()):
        if preferred in names:
            return preferred
    return sorted(names)[0]


def loss_composition(
    epochs: np.ndarray,
    series: dict[str, np.ndarray],
    roles: dict[str, list[str]],
) -> dict:
    physical_name = preferred_metric(roles, "physical_loss")
    reconstruction_name = preferred_metric(roles, "reconstruction_loss")
    tv_name = preferred_metric(roles, "tv_loss")
    if not all((physical_name, reconstruction_name, tv_name)):
        return {
            "status": "UNAVAILABLE",
            "reason": "physical, reconstruction, and TV histories are all required",
            "available_inputs": {
                "physical": physical_name,
                "reconstruction": reconstruction_name,
                "tv": tv_name,
            },
        }
    physical = series[physical_name]
    reconstruction = series[reconstruction_name]
    tv = series[tv_name]
    final_finite = np.flatnonzero(np.isfinite(physical))
    if final_finite.size == 0:
        return {"status": "UNAVAILABLE", "reason": "physical loss has no finite values"}
    targets = [*COMPOSITION_TARGETS, int(epochs[final_finite[-1]])]
    rows = []
    for target in dict.fromkeys(targets):
        valid = np.flatnonzero(
            np.isfinite(physical) & np.isfinite(reconstruction) & np.isfinite(tv)
        )
        if valid.size == 0:
            break
        index = min(valid.tolist(), key=lambda idx: (abs(epochs[idx] - target), epochs[idx]))
        denominator = float(physical[index])
        rows.append({
            "target_epoch": int(target),
            "recorded_epoch": json_number(epochs[index]),
            "physical_loss": float(physical[index]),
            "reconstruction_loss": float(reconstruction[index]),
            "tv_loss": float(tv[index]),
            "reconstruction_over_physical": (
                float(reconstruction[index] / denominator) if denominator != 0 else None
            ),
            "tv_over_physical": float(tv[index] / denominator) if denominator != 0 else None,
        })
    return {
        "status": "AVAILABLE",
        "metric_names": {
            "physical": physical_name,
            "reconstruction": reconstruction_name,
            "tv": tv_name,
        },
        "rows": rows,
        "causal_interpretation": "not asserted; ratios are descriptive only",
    }


def detect_snapshot_matrix(history: dict, length: int) -> tuple[str, np.ndarray] | None:
    for key, value in history.items():
        lowered = key.lower()
        if not any(token in lowered for token in ("x_snapshot", "x_hat", "foreground_mean_x", "abundance_vector")):
            continue
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if matrix.shape == (length, EXPECTED_CANDIDATES):
            return key, matrix
    return None


def detect_support_matrix(history: dict, length: int) -> tuple[str, np.ndarray] | None:
    for key, value in history.items():
        if "support" not in key.lower():
            continue
        try:
            matrix = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if matrix.shape == (length, EXPECTED_CANDIDATES) and matrix.dtype == np.bool_:
            return key, matrix
    return None


def earliest_stable_support(epochs: np.ndarray, support: np.ndarray) -> float | None:
    final = support[-1]
    for index in range(len(support)):
        if np.all(support[index:] == final):
            return json_number(epochs[index])
    return None


def support_flips_after(
    epochs: np.ndarray, support: np.ndarray, threshold: int
) -> dict:
    if len(support) < 2:
        return {"candidate_membership_flips": 0, "transitions_with_any_flip": 0}
    changes = support[1:] != support[:-1]
    selected = epochs[1:] > threshold
    return {
        "candidate_membership_flips": int(changes[selected].sum()),
        "transitions_with_any_flip": int(np.any(changes[selected], axis=1).sum()),
    }


def x_support_stability(history: dict, epochs: np.ndarray) -> dict:
    snapshot = detect_snapshot_matrix(history, len(epochs))
    direct_support = detect_support_matrix(history, len(epochs))
    if snapshot is None and direct_support is None:
        return {
            "status": "HISTORICAL_SUPPORT_STABILITY_UNAVAILABLE",
            "reason": "no checkpoint-aligned full X vectors or support snapshots",
        }
    result = {"status": "AVAILABLE"}
    if snapshot is not None:
        name, matrix = snapshot
        final = matrix[-1]
        denominator = max(float(np.linalg.norm(final)), 1.0e-30)
        distances = np.linalg.norm(matrix - final[None, :], axis=1) / denominator
        distance_bands = {}
        for band in X_DISTANCE_BANDS:
            eligible = [
                index for index in range(len(distances))
                if np.all(distances[index:] < band)
            ]
            distance_bands[f"{100 * band:g}%"] = (
                json_number(epochs[eligible[0]]) if eligible else None
            )
        result.update({
            "x_snapshot_field": name,
            "relative_l2_distance_to_final": [json_number(value) for value in distances],
            "earliest_epoch_remaining_below_distance": distance_bands,
        })
        support_sources = {
            f"{gate:g}": matrix > gate for gate in SUPPORT_GATES
        }
    else:
        name, support = direct_support
        result["support_snapshot_field"] = name
        support_sources = {"historical_boolean_support": support}
    support_results = {}
    for gate, support in support_sources.items():
        support_results[gate] = {
            "final_support": np.flatnonzero(support[-1]).astype(int).tolist(),
            "earliest_epoch_equal_to_final_and_never_changes": earliest_stable_support(epochs, support),
            "flips_after_epoch": {
                str(threshold): support_flips_after(epochs, support, threshold)
                for threshold in (800, 1000, 1500, 2000)
            },
        }
    result["support_by_gate"] = support_results
    return result


def audit_case(case: str, paths: dict) -> tuple[dict, list[dict]]:
    history = read_json(Path(paths["history_path"]))
    report = read_json(Path(paths["report_path"])) if paths["report_path"] else {}
    epochs = history_epochs(history)
    schema = {
        "case": case,
        "history_path": paths["history_path"],
        "report_path": paths["report_path"],
        "history_fields": {key: value_schema(value) for key, value in history.items()},
        "report_fields": {key: value_schema(value) for key, value in report.items()},
        "cadence": cadence_summary(epochs),
    }
    if epochs is None:
        return {
            "case": case,
            "status": "HISTORY_SCHEMA_INSUFFICIENT",
            "schema": schema,
            "available_metrics": [],
            "time_to_final_band": {},
            "reconstruction_progress": [],
            "total_physical_tv_progress": {},
            "late_loss_composition": {"status": "UNAVAILABLE"},
            "support_x_stability": {"status": "HISTORICAL_SUPPORT_STABILITY_UNAVAILABLE"},
        }, []

    series = extract_scalar_series(history, report, epochs)
    roles = metric_roles(sorted(series))
    schema.update({
        "available_scalar_metrics": sorted(series),
        "detected_metric_roles": roles,
        "support_or_x_snapshot_fields": [
            key for key in history
            if "support" in key.lower() or "snapshot" in key.lower()
        ],
    })
    band_summary = {}
    convergence_rows = []
    for metric, values in sorted(series.items()):
        band_summary[metric] = {}
        for band in FINAL_BANDS:
            result = earliest_final_band(epochs, values, band)
            label = f"{100 * band:g}%"
            band_summary[metric][label] = result
            convergence_rows.append({
                "case": case,
                "metric": metric,
                "band_percent": 100 * band,
                **result,
            })

    progress = {}
    for role in ("reconstruction_loss", "total_loss", "physical_loss", "tv_loss"):
        metric = preferred_metric(roles, role)
        progress[role] = {
            "metric": metric,
            "rows": progress_table(epochs, series[metric], metric) if metric else [],
        }
    final_epoch = report.get("final_training_epoch", epochs[-1])
    stability = x_support_stability(history, epochs)
    case_summary = {
        "case": case,
        "status": "AUDITED",
        "final_epoch": json_number(final_epoch),
        "recorded_cadence": schema["cadence"],
        "available_metrics": sorted(series),
        "metric_roles": roles,
        "time_to_final_band": band_summary,
        "reconstruction_progress": progress["reconstruction_loss"]["rows"],
        "total_physical_tv_progress": {
            role: progress[role] for role in ("total_loss", "physical_loss", "tv_loss")
        },
        "late_loss_composition": loss_composition(epochs, series, roles),
        "support_x_stability": stability,
        "why_training_continued_to_final_epoch": {
            "stop_reason_from_report": report.get("stop_reason", "UNAVAILABLE"),
            "final_stable_checks_from_history": (
                history.get("stable_checks", [None])[-1]
                if isinstance(history.get("stable_checks"), list) and history.get("stable_checks")
                else None
            ),
            "interpretation": "descriptive fields only; no causal claim or new stopping rule",
        },
        "schema": schema,
    }
    return case_summary, convergence_rows


def overall_status(case_summaries: dict) -> tuple[str, str]:
    if set(case_summaries) != set(REQUIRED_CASES):
        return (
            "HISTORY_INSUFFICIENT_NEEDS_SEGMENTED_PILOT",
            "one or more required K=1/K=3/K=5 histories or reports are unavailable",
        )
    sufficient = all(
        summary.get("support_x_stability", {}).get("status") == "AVAILABLE"
        and summary.get("metric_roles", {}).get("reconstruction_loss")
        and summary.get("metric_roles", {}).get("physical_loss")
        for summary in case_summaries.values()
    )
    if sufficient:
        return (
            "HISTORY_SUFFICIENT_FOR_FASTSTOP_DESIGN",
            "all cases contain reconstruction/physical loss and checkpoint-aligned support or X histories",
        )
    return (
        "HISTORY_INSUFFICIENT_NEEDS_SEGMENTED_PILOT",
        "existing histories do not jointly establish truth-independent reconstruction and support/X stability",
    )


def write_convergence_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "case", "metric", "band_percent", "status", "earliest_epoch",
        "value_at_earliest_epoch", "final_epoch", "final_value",
        "absolute_deviation_at_earliest_epoch", "absolute_tolerance",
        "relative_to_final_numerically_unstable_near_zero",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_audit(inventory: dict, input_root: Path, output_dir: Path) -> dict:
    case_summaries = {}
    convergence_rows = []
    schema_cases = {}
    errors = {}
    for case, paths in inventory.items():
        if paths["history_path"] is None or paths["report_path"] is None:
            continue
        try:
            summary, rows = audit_case(case, paths)
        except (OSError, ValueError, RuntimeError) as error:
            errors[case] = str(error)
            continue
        case_summaries[case] = summary
        schema_cases[case] = summary["schema"]
        convergence_rows.extend(rows)
    status, reason = overall_status(case_summaries)
    history_schema = {
        "input_scope": str(input_root),
        "discovery_depth": "v50 root and immediate case directories only",
        "inventory": inventory,
        "cases": schema_cases,
        "errors": errors,
    }
    summary = {
        "status": status,
        "status_reason": reason,
        "descriptive_only": True,
        "new_stopping_rule_defined": False,
        "truth_used_for_faststop_design": False,
        "cases": case_summaries,
        "missing_cases": [case for case in REQUIRED_CASES if case not in case_summaries],
        "history_sufficient_to_propose_truth_independent_faststop_diagnostic": status == "HISTORY_SUFFICIENT_FOR_FASTSTOP_DESIGN",
    }
    atomic_write_json(output_dir / "history_schema.json", history_schema)
    write_convergence_csv(output_dir / "convergence_table.csv", convergence_rows)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inventory immediate v50 histories and schemas without writing audit outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    inventory = discover_inputs(input_root)
    found_fields = {}
    support_snapshots = {}
    for case, paths in inventory.items():
        history_path = paths.get("history_path")
        if not history_path:
            found_fields[case] = []
            support_snapshots[case] = False
            continue
        try:
            history = read_json(Path(history_path))
        except (OSError, ValueError, RuntimeError):
            found_fields[case] = []
            support_snapshots[case] = False
            continue
        found_fields[case] = sorted(history)
        epochs = history_epochs(history)
        support_snapshots[case] = bool(
            epochs is not None
            and (
                detect_snapshot_matrix(history, len(epochs)) is not None
                or detect_support_matrix(history, len(epochs)) is not None
            )
        )
    if args.dry_run:
        complete = all(
            row["history_path"] is not None and row["report_path"] is not None
            for row in inventory.values()
        )
        print(json.dumps({
            "status": "DRY_RUN_READY" if complete else "DRY_RUN_READY_HISTORY_UNAVAILABLE_LOCALLY",
            "input_root": str(input_root),
            "inventory": inventory,
            "actual_history_fields": found_fields,
            "support_or_x_snapshots_exist": support_snapshots,
            "full_audit_must_run_where_v50_results_exist": not complete,
            "training_performed": False,
            "outputs_written": False,
            "epoch_recommendation_made": False,
        }, ensure_ascii=False, indent=2))
        return
    summary = run_audit(inventory, input_root, args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
