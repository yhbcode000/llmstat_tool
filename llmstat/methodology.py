"""Core statistical methodology — calibrated LLM estimator pipeline.

Implements the paper's workflow:
  1. Calibration: estimate b̂, λ̂, δ̂ from paired human-LLM data
  2. Feasibility: check (|b̂| + δ̂·Δ̂_max)/Δ < 1 and λ̂ ≤ 10
  3. N_LLM: compute adjusted sample size via Eq. (llm-n)
  4. Cost: compare human-only vs LLM-augmented designs
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats as sp_stats


class CalibrationResult(NamedTuple):
    """Calibration estimates from paired human-LLM data."""

    bias: float  # b̂  — mean(LLM − human)
    se_bias: float  # SE(b̂)
    lambda_hat: float  # λ̂ = Var(LLM) / Var(human)
    delta_hat: float  # δ̂ — prompt-condition misclassification rate
    delta_max: float  # Δ̂_max — estimated condition range


class FeasibilityResult(NamedTuple):
    """Result of the feasibility check."""

    feasible: bool
    penalty: float  # ρ_bias = (|b̂| + δ̂·Δ̂_max) / Δ
    reason: str  # human-readable explanation


class DesignResult(NamedTuple):
    """Adjusted design parameters."""

    n_llm: int  # N_LLM per condition
    human_cost: float
    llm_cost: float
    reduction: float  # 1 − C_LLM / C_human (as fraction)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    human_responses: np.ndarray,  # shape (M_c, K)
    llm_responses: np.ndarray,    # shape (M_c, K)
    condition_labels: np.ndarray,  # shape (M_c,) — ground truth condition
    prompt_labels: np.ndarray,     # shape (M_v,) — ϕ(s) from validation prompts
    prompt_truth: np.ndarray,      # shape (M_v,) — true condition for each prompt
) -> CalibrationResult:
    """Estimate b̂, λ̂, δ̂ from paired calibration data.

    Parameters
    ----------
    human_responses : (M_c, K) array
        Human responses for each calibration observation × condition.
    llm_responses : (M_c, K) array
        LLM responses for each calibration observation × condition.
    condition_labels : (M_c,) array
        Ground-truth condition index for each calibration observation.
    prompt_labels : (M_v,) array
        Mapped condition from validation prompts.
    prompt_truth : (M_v,) array
        True condition for each validation prompt.

    Returns
    -------
    CalibrationResult
    """
    K = human_responses.shape[1]
    diffs = llm_responses - human_responses  # (M_c, K)

    # Per-condition bias
    bias_per_cond = np.array([np.mean(diffs[condition_labels == k, k]) for k in range(K)])
    se_per_cond = np.array([
        np.std(diffs[condition_labels == k, k], ddof=1) / math.sqrt(max(np.sum(condition_labels == k), 1))
        for k in range(K)
    ])

    # Overall bias (weighted by condition prevalence)
    cond_counts = np.array([np.sum(condition_labels == k) for k in range(K)])
    weights = cond_counts / cond_counts.sum()
    bias = float(np.average(bias_per_cond, weights=weights))
    se_bias = float(np.sqrt(np.sum((weights**2) * (se_per_cond**2))))

    # Variance ratio λ̂ = Var(LLM) / Var(human)
    human_var = np.var(human_responses, ddof=1) if human_responses.size > 1 else 1.0
    llm_var = np.var(llm_responses, ddof=1) if llm_responses.size > 1 else 1.0
    lambda_hat = float(llm_var / human_var) if human_var > 1e-12 else 1.0

    # Identifiability error δ̂
    matches = (prompt_labels == prompt_truth).sum()
    delta_hat = 1.0 - float(matches) / max(len(prompt_truth), 1)

    # Estimated condition range
    delta_max = float(np.max(human_responses.mean(axis=0)) - np.min(human_responses.mean(axis=0)))

    return CalibrationResult(
        bias=bias,
        se_bias=se_bias,
        lambda_hat=lambda_hat,
        delta_hat=delta_hat,
        delta_max=delta_max,
    )


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def check_feasibility(
    cal: CalibrationResult,
    min_effect: float,  # Δ — minimum effect of scientific interest
) -> FeasibilityResult:
    """Apply the feasibility rule from the paper.

    Feasible iff:
      ρ = (|b̂| + δ̂·Δ̂_max) / Δ  <  1   AND   λ̂ ≤ 10
    """
    penalty = (abs(cal.bias) + cal.delta_hat * cal.delta_max) / min_effect
    lambda_ok = cal.lambda_hat <= 10.0
    penalty_ok = penalty < 1.0

    if not lambda_ok:
        return FeasibilityResult(False, penalty, f"λ̂={cal.lambda_hat:.2f} > 10")
    if not penalty_ok:
        return FeasibilityResult(
            False, penalty,
            f"ρ={penalty:.3f} ≥ 1 (|b̂|={abs(cal.bias):.4f}, δ̂·Δ̂_max={cal.delta_hat * cal.delta_max:.4f}, Δ={min_effect})"
        )
    return FeasibilityResult(True, penalty, "feasible")


# ---------------------------------------------------------------------------
# Adjusted sample size
# ---------------------------------------------------------------------------

def compute_design(
    cal: CalibrationResult,
    min_effect: float,         # Δ
    n_human: int,              # N_h — baseline human per-condition sample
    cost_human: float,         # c_h — cost per human response
    cost_llm: float,           # c_LLM — cost per LLM response
    n_calibration: int,        # M_c — calibration sample size
    n_conditions: int = 2,     # K
) -> DesignResult:
    """Compute adjusted LLM sample size and cost comparison.

    Uses Eq. (llm-n) from the paper:
      N_LLM = ⌈ N_h · λ̂ / (1 − ρ_bias)² ⌉
    """
    penalty = (abs(cal.bias) + cal.delta_hat * cal.delta_max) / min_effect

    if penalty >= 1.0:
        # Infeasible — return raw human-only cost
        human_cost = n_conditions * n_human * cost_human
        return DesignResult(
            n_llm=n_human,
            human_cost=human_cost,
            llm_cost=human_cost,
            reduction=0.0,
        )

    n_llm = math.ceil(n_human * cal.lambda_hat / (1.0 - penalty) ** 2)
    human_cost = n_conditions * n_human * cost_human
    llm_cost = n_calibration * cost_human + n_conditions * n_llm * cost_llm
    reduction = 1.0 - llm_cost / human_cost if human_cost > 0 else 0.0

    return DesignResult(
        n_llm=n_llm,
        human_cost=human_cost,
        llm_cost=llm_cost,
        reduction=reduction,
    )


# ---------------------------------------------------------------------------
# Response generation helpers
# ---------------------------------------------------------------------------

def build_condition_prompts(
    conditions: list[str],
    outcome_description: str,
    n_per_condition: int,
) -> list[tuple[str, str]]:
    """Build LLM prompts for generating calibrated condition responses.

    Returns list of (system_prompt, user_prompt) tuples.
    """
    system = (
        "You are a research participant in a quantitative human-subject study. "
        "Respond ONLY with a single number on the specified scale. "
        "Do not include any explanation, units, or extra text."
    )
    prompts: list[tuple[str, str]] = []
    for cond in conditions:
        user = (
            f"Condition: {cond}\n"
            f"Outcome: {outcome_description}\n"
            f"Please provide your response as a single number."
        )
        for _ in range(n_per_condition):
            prompts.append((system, user))
    return prompts


def parse_numeric_response(text: str, default: float = 0.0) -> float:
    """Extract a numeric value from an LLM response string."""
    import re

    text = text.strip()
    # Try to find a standalone number (possibly with decimals and sign)
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        return float(match.group())
    return default
