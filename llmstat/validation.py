"""TOST (Two One-Sided Test) equivalence validation.

Implements the validation step from the paper's calibration protocol.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


class TOSTResult:
    """Result of a TOST equivalence test."""

    def __init__(
        self,
        equivalent: bool,
        p_value: float,
        statistic_upper: float,
        statistic_lower: float,
        critical_value: float,
        margin: float,
        estimate_diff: float,
        se_diff: float,
    ) -> None:
        self.equivalent = equivalent
        self.p_value = p_value
        self.statistic_upper = statistic_upper
        self.statistic_lower = statistic_lower
        self.critical_value = critical_value
        self.margin = margin
        self.estimate_diff = estimate_diff
        self.se_diff = se_diff

    def __repr__(self) -> str:
        status = "EQUIVALENT" if self.equivalent else "NOT EQUIVALENT"
        return (
            f"TOSTResult({status}, diff={self.estimate_diff:.4f}, "
            f"SE={self.se_diff:.4f}, p={self.p_value:.4f}, margin={self.margin})"
        )


def tost_two_sample(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> TOSTResult:
    """Two-sample TOST for equivalence of means.

    H₀: |μ_A − μ_B| ≥ margin   vs   H₁: |μ_A − μ_B| < margin

    Uses Welch's t-test (does not assume equal variances).
    """
    n_a, n_b = len(sample_a), len(sample_b)
    mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
    var_a, var_b = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1)

    diff = mean_a - mean_b
    se = np.sqrt(var_a / n_a + var_b / n_b)

    if se < 1e-15:
        # Degenerate case — no variance
        equivalent = abs(diff) < margin
        return TOSTResult(
            equivalent=equivalent,
            p_value=0.0 if equivalent else 1.0,
            statistic_upper=float("inf"),
            statistic_lower=float("-inf"),
            critical_value=sp_stats.t.ppf(1 - alpha, df=n_a + n_b - 2),
            margin=margin,
            estimate_diff=diff,
            se_diff=0.0,
        )

    # Welch degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
    df = num / denom if denom > 1e-15 else n_a + n_b - 2

    # Upper bound test: H₀¹: μ_A − μ_B ≤ −margin
    t_upper = (diff + margin) / se
    p_upper = 1.0 - sp_stats.t.cdf(t_upper, df)

    # Lower bound test: H₀²: μ_A − μ_B ≥ +margin
    t_lower = (diff - margin) / se
    p_lower = sp_stats.t.cdf(t_lower, df)

    # TOST rejects H₀ iff both one-sided tests reject
    p_value = max(p_upper, p_lower)
    critical = sp_stats.t.ppf(1 - alpha, df)
    equivalent = (t_upper > critical) and (t_lower < -critical)

    return TOSTResult(
        equivalent=equivalent,
        p_value=float(p_value),
        statistic_upper=float(t_upper),
        statistic_lower=float(t_lower),
        critical_value=float(critical),
        margin=margin,
        estimate_diff=float(diff),
        se_diff=float(se),
    )


def tost_one_sample(
    sample: np.ndarray,
    reference_mean: float,
    margin: float,
    alpha: float = 0.05,
) -> TOSTResult:
    """One-sample TOST for equivalence of a sample mean to a reference value.

    H₀: |μ − μ_ref| ≥ margin   vs   H₁: |μ − μ_ref| < margin
    """
    return tost_two_sample(
        sample_a=sample,
        sample_b=np.full_like(sample, reference_mean),
        margin=margin,
        alpha=alpha,
    )
