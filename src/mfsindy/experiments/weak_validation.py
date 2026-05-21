"""Held-out weak-form validation helpers for PDE-style examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeakValidationBlock:
    """One held-out weak-form block used for candidate scoring."""

    theta: np.ndarray
    rhs: np.ndarray
    group: str = "validation"
    trajectory: int = 0
    block: int = 0


def weak_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute an R^2 score for weak-form targets over all dimensions."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}.")
    if not np.all(np.isfinite(y_pred)):
        return -1e12

    residual = y_true - y_pred
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    denom = float(np.sum(centered**2))
    if denom <= 0.0:
        return -1e12
    return float(1.0 - np.sum(residual**2) / denom)


def evaluate_weak_form_models(
    coefficient_map: Mapping[str, np.ndarray],
    validation_blocks: Sequence[WeakValidationBlock],
    *,
    metric_name: str = "weak_r2",
    source_col: str = "model",
) -> pd.DataFrame:
    """Score coefficient maps on held-out weak-form blocks."""

    rows: list[dict[str, Any]] = []
    for block in validation_blocks:
        theta = np.asarray(block.theta, dtype=float)
        rhs = np.asarray(block.rhs, dtype=float)
        for model_name, coefficients in coefficient_map.items():
            coef = np.asarray(coefficients, dtype=float)
            try:
                pred = theta @ coef.T
                score = weak_r2_score(rhs, pred)
            except Exception:
                score = -1e12
            rows.append(
                {
                    source_col: model_name,
                    "group": block.group,
                    "trajectory": block.trajectory,
                    "block": block.block,
                    "metric": metric_name,
                    "value": float(score),
                }
            )
    return pd.DataFrame(rows)


def split_spatiotemporal_trajectory(
    trajectory: np.ndarray,
    t_grid: np.ndarray,
    *,
    time_axis: int | None = None,
    validation_fraction: float = 0.2,
    overlap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a spatiotemporal trajectory along its time axis."""

    trajectory = np.asarray(trajectory)
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be one-dimensional.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1).")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    if time_axis is None:
        candidates = [axis for axis, size in enumerate(trajectory.shape) if size == t_grid.shape[0]]
        if not candidates:
            raise ValueError(
                "Could not infer the time axis: no trajectory dimension matches len(t_grid)."
            )
        time_axis = candidates[0]

    time_axis = int(time_axis)
    if time_axis < 0:
        time_axis += trajectory.ndim
    if not 0 <= time_axis < trajectory.ndim:
        raise ValueError(f"Invalid time_axis {time_axis} for shape {trajectory.shape}.")
    if trajectory.shape[time_axis] != t_grid.shape[0]:
        raise ValueError("trajectory and t_grid must agree along the time axis.")

    n_total = trajectory.shape[time_axis]
    n_val = max(2, int(np.ceil(n_total * validation_fraction)))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val
    val_start = max(0, n_train - overlap)

    train_slices = [slice(None)] * trajectory.ndim
    val_slices = [slice(None)] * trajectory.ndim
    train_slices[time_axis] = slice(0, n_train)
    val_slices[time_axis] = slice(val_start, None)

    train = trajectory[tuple(train_slices)].copy()
    val = trajectory[tuple(val_slices)].copy()
    train_t = t_grid[:n_train] - t_grid[0]
    val_t = t_grid[val_start:] - t_grid[val_start]
    return train, val, train_t, val_t


def get_library_feature_names(
    library: Any,
    reference_data: np.ndarray,
    *,
    input_features: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Best-effort feature-name extraction for a fitted SINDy library."""

    try:
        library.fit([reference_data])
    except Exception:
        try:
            library.fit(reference_data)
        except Exception:
            return ()

    getter = getattr(library, "get_feature_names", None)
    if getter is None:
        return ()

    for kwargs in (
        {"input_features": list(input_features)} if input_features is not None else None,
        {},
    ):
        if kwargs is None:
            continue
        try:
            names = getter(**kwargs)
            return tuple(str(name) for name in names)
        except Exception:
            continue

    try:
        names = getter(list(input_features)) if input_features is not None else getter()
        return tuple(str(name) for name in names)
    except Exception:
        return ()


def format_coefficient_equations(
    coefficient_map: Mapping[str, np.ndarray],
    *,
    feature_names: Sequence[str],
    state_names: Sequence[str],
    precision: int = 3,
    tol: float = 1e-10,
) -> str:
    """Format a map of coefficient matrices as readable equations."""

    feature_names = tuple(feature_names)
    state_names = tuple(state_names)
    sections: list[str] = []
    for model_name, coefficients in coefficient_map.items():
        coef = np.asarray(coefficients, dtype=float)
        if coef.ndim == 1:
            coef = coef[None, :]
        local_features = feature_names or tuple(f"term_{j}" for j in range(coef.shape[1]))
        lines = [model_name]
        for state_name, row in zip(state_names, coef, strict=True):
            terms: list[tuple[str, str]] = []
            for value, feature_name in zip(row, local_features, strict=True):
                value = float(value)
                if abs(value) <= tol:
                    continue
                sign = "-" if value < 0 else "+"
                magnitude = abs(value)
                if feature_name == "1":
                    term = f"{magnitude:.{precision}g}"
                else:
                    term = f"{magnitude:.{precision}g} {feature_name}"
                terms.append((sign, term))

            if not terms:
                rhs = "0"
            else:
                first_sign, first_term = terms[0]
                rhs_parts = [first_term if first_sign == "+" else f"-{first_term}"]
                rhs_parts.extend(f" {sign} {term}" for sign, term in terms[1:])
                rhs = "".join(rhs_parts)
            lines.append(f"{state_name} = {rhs}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


__all__ = [
    "WeakValidationBlock",
    "evaluate_weak_form_models",
    "format_coefficient_equations",
    "get_library_feature_names",
    "split_spatiotemporal_trajectory",
    "weak_r2_score",
]
