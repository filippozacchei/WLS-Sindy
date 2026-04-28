"""Grid-search utilities for selecting experiment hyperparameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm 


@dataclass(frozen=True)
class MetricTerm:
    """One additive term in a grid-search objective."""

    source: str
    metric: str
    weight: float = 1.0


@dataclass
class HyperparameterSearchResult:
    """Summary of a completed hyperparameter grid search."""

    best_params: dict[str, Any]
    best_score: float
    best_config: Any
    best_result: Any
    results: pd.DataFrame


def _clone_with_updates(base_config: Any, updates: Mapping[str, Any]) -> Any:
    cfg = deepcopy(base_config)
    missing = [name for name in updates if not hasattr(cfg, name)]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise AttributeError(f"Unknown hyperparameter(s) for config: {missing_str}")
    for name, value in updates.items():
        setattr(cfg, name, value)
    return cfg


def _coerce_results_frame(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], pd.DataFrame):
        return result[0]
    raise TypeError(
        "Expected `evaluate` to return a pandas DataFrame or a tuple whose first element "
        "is a pandas DataFrame."
    )


def make_metric_scorer(
    *,
    source: str,
    metric: str,
    source_col: str = "model",
    reducer: Callable[[pd.Series], float] = np.mean,
) -> Callable[[Any], float]:
    """Build a scorer from the long-format error DataFrame returned by experiment helpers."""

    def score(result: Any) -> float:
        df = _coerce_results_frame(result)
        mask = (df[source_col] == source) & (df["metric"] == metric)
        if not mask.any():
            raise ValueError(
                f"No rows found for source={source!r}, metric={metric!r}, source_col={source_col!r}."
            )
        values = df.loc[mask, "value"]
        return float(reducer(values))

    return score


def make_combined_metric_scorer(
    terms: Sequence[MetricTerm],
    *,
    source_col: str = "model",
    reducer: Callable[[pd.Series], float] = np.mean,
) -> Callable[[Any], float]:
    """Build a weighted additive scorer from multiple metric terms."""

    if not terms:
        raise ValueError("At least one MetricTerm is required.")

    def score(result: Any) -> float:
        df = _coerce_results_frame(result)
        total = 0.0
        for term in terms:
            mask = (df[source_col] == term.source) & (df["metric"] == term.metric)
            if not mask.any():
                raise ValueError(
                    "No rows found for "
                    f"source={term.source!r}, metric={term.metric!r}, source_col={source_col!r}."
                )
            values = df.loc[mask, "value"]
            total += float(term.weight) * float(reducer(values))
        return float(total)

    return score


def optimize_hyperparams(
    base_config: Any,
    *,
    param_grid: Mapping[str, Sequence[Any]],
    evaluate: Callable[[Any], Any],
    score_fn: Callable[[Any], float],
    maximize: bool = False,
    raise_on_error: bool = False,
) -> HyperparameterSearchResult:
    """Run a simple grid search over config fields.

    Parameters
    ----------
    base_config
        Configuration object to clone for each grid point.
    param_grid
        Mapping from config attribute name to candidate values, for example
        ``{"stlsq_threshold": [0.01, 0.05], "poly_degree": [1, 2], "H_xt": [0.05, 0.1]}``.
    evaluate
        Callable receiving the updated config and returning an experiment result.
    score_fn
        Maps the output of ``evaluate`` to a scalar score.
    maximize
        If ``True``, larger scores are better. Otherwise scores are minimized.
    raise_on_error
        If ``False``, failed combinations are recorded and the search continues.
    """

    if not param_grid:
        raise ValueError("param_grid must contain at least one hyperparameter.")

    grid_names = list(param_grid)
    grid_values = []
    for name in grid_names:
        values = list(param_grid[name])
        if not values:
            raise ValueError(f"param_grid[{name!r}] must contain at least one value.")
        grid_values.append(values)

    rows: list[dict[str, Any]] = []
    best_score: float | None = None
    best_params: dict[str, Any] | None = None
    best_config: Any = None
    best_result: Any = None

    for trial_index, combo in tqdm(enumerate(product(*grid_values))):
        params = dict(zip(grid_names, combo))
        cfg = _clone_with_updates(base_config, params)

        row = {"trial": trial_index, **params}
        try:
            result = evaluate(cfg)
            score = float(score_fn(result))
            row.update({"status": "ok", "score": score, "error": ""})

            is_better = best_score is None
            if best_score is not None:
                is_better = score > best_score if maximize else score < best_score

            if is_better:
                best_score = score
                best_params = dict(params)
                best_config = cfg
                best_result = result
        except Exception as exc:
            if raise_on_error:
                raise
            row.update(
                {
                    "status": "failed",
                    "score": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        rows.append(row)

    if best_params is None or best_score is None:
        raise RuntimeError("Hyperparameter search finished with no successful grid points.")

    results = pd.DataFrame(rows)
    results["_status_rank"] = (results["status"] != "ok").astype(int)
    results = (
        results.sort_values(
            by=["_status_rank", "score"],
            ascending=[True, not maximize],
            na_position="last",
        )
        .drop(columns="_status_rank")
        .reset_index(drop=True)
    )

    return HyperparameterSearchResult(
        best_params=best_params,
        best_score=best_score,
        best_config=best_config,
        best_result=best_result,
        results=results,
    )


__all__ = [
    "HyperparameterSearchResult",
    "MetricTerm",
    "make_combined_metric_scorer",
    "make_metric_scorer",
    "optimize_hyperparams",
]
