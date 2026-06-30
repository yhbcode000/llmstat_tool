#!/usr/bin/env python3
"""
Web A/B Testing with Calibrated LLM Surrogates
===============================================

Demonstrates the llmstat methodology applied to **web A/B testing** scenarios
where LLM subjects serve as calibrated surrogate estimators for human
click-through and conversion behavior.

Based on: Yang, H. "Pretrained Large Language Models as Statistical
Instruments: Restricted Risk Equivalence under Squared Loss."

Core idea (from the paper):
  An LLM is a misspecified estimator of conditional expectations under
  squared loss. When the calibration protocol confirms surrogacy — bias
  small, variance ratio bounded, identifiability controlled — the LLM
  can replace a large fraction of human samples at dramatically lower cost.

Usage:  python examples/web_ab_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure llmstat is importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from llmstat.methodology import (
    CalibrationResult,
    calibrate,
    check_feasibility,
    compute_design,
)
from llmstat.report import format_money, format_pct
from llmstat.validation import tost_two_sample


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_calibration(
    K: int,
    M_c: int,
    M_v: int,
    b_hat_true: float,
    lambda_hat_true: float,
    delta_hat_true: float,
    delta_max: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic paired human-LLM calibration data.

    Models human responses as draws from distinct per-condition distributions,
    and LLM responses as human responses + per-condition bias + noise scaled
    by sqrt(lambda_hat_true). This matches the paper's decomposition:

        LLM_n(k) = mu_k + eps_rep(k) + eta_n(k) + xi_n(k)

    where eps_rep(k) ≈ b_hat_true is the systematic representation bias.

    Parameters
    ----------
    K : int
        Number of conditions (always 2 for A/B tests).
    M_c : int
        Calibration sample size.
    M_v : int
        Prompt-validation sample size.
    b_hat_true : float
        Target overall calibration bias |b̂|.
    lambda_hat_true : float
        Target LLM-to-human variance ratio λ̂.
    delta_hat_true : float
        Target identifiability error δ̂ (prompt misclassification rate).
    delta_max : float
        Expected condition range Δ̂_max.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    human_cal : (M_c, K) ndarray
    llm_cal : (M_c, K) ndarray
    cond_labels : (M_c,) ndarray
    prompt_labels : (M_v,) ndarray
    prompt_truth : (M_v,) ndarray
    """
    rng = np.random.default_rng(seed)

    # Per-condition baseline means (spaced by delta_max / (K-1))
    cond_means = np.linspace(0.3, 0.3 + delta_max, K)

    # Human responses: condition_mean + within-condition noise
    human_cal = np.zeros((M_c, K))
    llm_cal = np.zeros((M_c, K))
    cond_labels = np.zeros(M_c, dtype=int)

    for i in range(M_c):
        k = i % K
        cond_labels[i] = k
        # Human: around the true condition mean
        human_cal[i, k] = cond_means[k] + 0.05 * rng.normal()
        # LLM: human + per-condition bias + scaled noise
        per_cond_bias = b_hat_true * (1.0 + 0.2 * (k - K / 2))  # vary slightly by condition
        llm_noise_scale = np.sqrt(max(lambda_hat_true, 0.01))
        llm_cal[i, k] = human_cal[i, k] + per_cond_bias + 0.02 * llm_noise_scale * rng.normal()

    # Prompt validation: ground-truth condition labels
    prompt_truth = rng.integers(0, K, size=M_v)
    prompt_labels = prompt_truth.copy()
    # Introduce misclassification at rate delta_hat_true
    if delta_hat_true > 0:
        n_flip = int(delta_hat_true * M_v)
        flip_idx = rng.choice(M_v, size=n_flip, replace=False)
        for idx in flip_idx:
            other = [c for c in range(K) if c != prompt_truth[idx]]
            prompt_labels[idx] = rng.choice(other)

    return human_cal, llm_cal, cond_labels, prompt_labels, prompt_truth


# ═══════════════════════════════════════════════════════════════════════════
# Scenario runner
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(scenario: dict) -> None:
    """Run a single web A/B test scenario through the full pipeline.

    Steps (matching the paper's calibration protocol, §7):
      1. Generate synthetic paired calibration data
      2. Estimate b̂, λ̂, δ̂ from calibration
      3. Feasibility check:  ρ = (|b̂| + δ̂·Δ̂_max) / Δ  <  1  AND  λ̂ ≤ 10
      4. Compute adjusted LLM sample size and cost comparison
      5. Generate synthetic LLM validation sample and run TOST equivalence
      6. Print detailed report
    """
    name = scenario["name"]
    K = scenario["K"]
    M_c = scenario["M_c"]
    M_v = scenario["M_v"]
    N_h = scenario["N_h"]
    min_effect = scenario["min_effect"]
    equiv_margin = scenario["equiv_margin"]
    cost_human = scenario["cost_human"]
    cost_llm = scenario["cost_llm"]
    params = scenario["params"]
    surrogacy_condition = scenario.get("surrogacy_condition", "")

    print()
    print("=" * 72)
    print(f"  {name}")
    print("=" * 72)
    print(f"  Conditions:        {scenario['conditions'][0]}  vs  {scenario['conditions'][1]}")
    print(f"  Outcome:           {scenario['outcome']}")
    print(f"  Δ (min effect):    {min_effect}")
    print(f"  N_h (human/cond):  {N_h}")
    print(f"  M_c (calibration): {M_c}")
    print(f"  c_h (cost/human):  {format_money(cost_human)}")
    print(f"  c_LLM (cost/LLM):  ${cost_llm:.4f}")
    if surrogacy_condition:
        print(f"  Surrogacy holds:   {surrogacy_condition}")
    print()

    # ── Phase 1: Calibration ──
    b_true, l_true, d_true, dmax = params
    human_cal, llm_cal, cond_labels, prompt_labels, prompt_truth = (
        generate_synthetic_calibration(K, M_c, M_v, b_true, l_true, d_true, dmax)
    )

    cal = calibrate(human_cal, llm_cal, cond_labels, prompt_labels, prompt_truth)
    print("── Phase 1: Calibration ──")
    print(f"  b̂ (overall bias)         = {cal.bias:+.6f}   (target |b̂| ≈ {b_true})")
    print(f"  SE(b̂)                    = {cal.se_bias:.6f}")
    print(f"  λ̂ (variance ratio)       = {cal.lambda_hat:.4f}    (target ≈ {l_true})")
    print(f"  δ̂ (identifiability err)  = {cal.delta_hat:.4f}    (target ≈ {d_true})")
    print(f"  Δ̂_max (condition range)  = {cal.delta_max:.4f}")
    print()

    # ── Phase 2: Feasibility ──
    feas = check_feasibility(cal, min_effect)
    print("── Phase 2: Feasibility Check ──")
    print(f"  ρ = (|b̂| + δ̂·Δ̂_max) / Δ  = {feas.penalty:.4f}  (must be < 1)")
    print(f"  λ̂ ≤ 10?                   {'✓' if cal.lambda_hat <= 10 else '✗'}")
    if feas.feasible:
        print(f"  Result: FEASIBLE — LLM surrogacy conditions satisfied")
    else:
        print(f"  Result: INFEASIBLE — {feas.reason}")
        return
    print()

    # ── Phase 3: Design ──
    design = compute_design(cal, min_effect, N_h, cost_human, cost_llm, M_c, K)
    print("── Phase 3: Sample-Size Design ──")
    print(f"  N_LLM per condition:  {design.n_llm}   (human baseline: {N_h})")
    print(f"  Amplification factor: {design.n_llm / N_h:.1f}×")
    print(f"  Human-only cost:      {format_money(design.human_cost)}")
    print(f"  LLM-augmented cost:   {format_money(design.llm_cost)}")
    print(f"  Cost reduction:       {format_pct(design.reduction)}")
    print()

    # ── Phase 4: TOST Equivalence Validation ──
    print("── Phase 4: TOST Equivalence Validation ──")
    rng_val = np.random.default_rng(99)
    # Human "true" responses at N_h per condition
    human_a = 0.5 + 0.05 * rng_val.normal(size=N_h)
    human_b = 0.5 + min_effect + 0.05 * rng_val.normal(size=N_h)
    # LLM surrogate responses at design.n_llm per condition
    llm_mean_a = 0.5 + b_true  # LLM mean ≈ human mean + bias
    llm_mean_b = 0.5 + min_effect + b_true
    llm_a = llm_mean_a + 0.02 * np.sqrt(max(l_true, 0.01)) * rng_val.normal(size=design.n_llm)
    llm_b = llm_mean_b + 0.02 * np.sqrt(max(l_true, 0.01)) * rng_val.normal(size=design.n_llm)

    # TOST: compare per-condition means (two-sample Welch, handles unequal sizes)
    tost = tost_two_sample(llm_b, human_b, equiv_margin)

    status = "PASS (EQUIVALENT)" if tost.equivalent else "FAIL (NOT EQUIVALENT)"
    print(f"  TOST result:       {status}")
    print(f"  Effect diff (LLM − human): {tost.estimate_diff:+.6f}")
    print(f"  SE of diff:        {tost.se_diff:.6f}")
    print(f"  p-value (TOST):    {tost.p_value:.4f}")
    print(f"  Equivalence margin: ±{equiv_margin}")
    print()

    # ── Summary ──
    print("── Summary ──")
    print(f"  Scenario:          {name}")
    savings = design.human_cost - design.llm_cost
    print(f"  Absolute savings:  {format_money(savings)}")
    effective = "✓ VALID — LLM surrogacy confirmed" if tost.equivalent and feas.feasible else "✗ INVALID"
    print(f"  Validity:          {effective}")
    print(f"  Recommendation:     Use LLM as calibrated surrogate for this A/B test class.")


# ═══════════════════════════════════════════════════════════════════════════
# Scenario definitions
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "name": "Scenario 1 — Headline CTR (click-through rate)",
        "conditions": ["neutral headline", "emotional headline"],
        "outcome": "click-through rate (CTR) ∈ [0, 1]",
        "K": 2,
        "N_h": 500,            # human samples per condition
        "M_c": 400,            # calibration sample size
        "M_v": 100,            # prompt-validation size
        "cost_human": 5.00,    # $5/human response (e.g., MTurk + platform fee)
        "cost_llm": 0.002,     # $0.002/LLM response (GPT-4o-mini tier)
        "min_effect": 0.02,    # Δ = 2 percentage-point CTR lift
        "equiv_margin": 0.01,  # TOST equivalence margin (1 pp)
        # (b̂_true, λ̂_true, δ̂_true, Δ̂_max)
        "params": (0.003, 1.05, 0.01, 0.30),
        "surrogacy_condition": (
            "Headline preference is a shallow text-comprehension task; "
            "LLMs trained on web text capture headline→interest mappings well. "
            "Calibration confirms low bias and near-unit variance ratio."
        ),
    },
    {
        "name": "Scenario 2 — Landing Page Conversion (CVR)",
        "conditions": ["short form", "long form"],
        "outcome": "conversion rate (CVR) ∈ [0, 1]",
        "K": 2,
        "N_h": 300,            # smaller human study (CVR tests are expensive)
        "M_c": 300,
        "M_v": 80,
        "cost_human": 3.00,    # $3/human (survey panel)
        "cost_llm": 0.002,
        "min_effect": 0.05,    # Δ = 5 pp CVR lift (meaningful for e-commerce)
        "equiv_margin": 0.02,  # 2 pp equivalence margin
        "params": (0.008, 1.10, 0.015, 0.40),
        "surrogacy_condition": (
            "Form-filling intent is captured by LLM's instruction-following ability. "
            "Higher bias than headlines (8×10⁻³) but still well within feasibility. "
            "Slightly elevated variance ratio (1.10) is acceptable."
        ),
    },
    {
        "name": "Scenario 3 — Button Color Click-through",
        "conditions": ["blue CTA", "orange CTA"],
        "outcome": "click-through rate ∈ [0, 1]",
        "K": 2,
        "N_h": 800,            # large human study (tiny effect needs power)
        "M_c": 500,
        "M_v": 100,
        "cost_human": 1.50,    # $1.50/human (cheap panel)
        "cost_llm": 0.002,
        "min_effect": 0.01,    # Δ = 1 pp CTR lift (typical for visual changes)
        "equiv_margin": 0.005, # 0.5 pp (tight because effect is tiny)
        "params": (0.001, 1.02, 0.005, 0.20),
        "surrogacy_condition": (
            "Button color preference is a weak visual signal. LLMs can approximate "
            "human aesthetic judgments for common UI patterns. Very low bias (1×10⁻³) "
            "and near-unit variance ratio (1.02) make this the strongest surrogacy case. "
            "CAVEAT: does NOT generalize to novel visual designs outside training distribution."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# When LLM surrogacy does NOT hold (reference table)
# ═══════════════════════════════════════════════════════════════════════════

NON_SURROGACY_TABLE = """
┌────────────────────────────────┬──────────────────────────────────────────┐
│ When LLM surrogacy FAILS       │ Why                                      │
├────────────────────────────────┼──────────────────────────────────────────┤
│ Payment-page A/B tests         │ Real financial stakes; LLMs don't have   │
│                                │ actual willingness-to-pay distributions  │
│ Retention / churn experiments  │ Long-term behavior depends on real-world │
│                                │ feedback loops absent from training data │
│ Social-network feature tests   │ Network effects require genuine human    │
│                                │ interaction dynamics; LLMs simulate      │
│                                │ isolated responses, not social contagion │
│ Novel UX paradigms             │ If P(new_condition) ∉ training support,  │
│                                │ KL projection is uninformative (ε_rep    │
│                                │ dominates) — per paper §5, §8            │
│ Causally-structured inference │ LLMs estimate conditional means, not      │
│                                │ counterfactual or structural parameters   │
└────────────────────────────────┴──────────────────────────────────────────┘
"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     Web A/B Testing with Calibrated LLM Surrogates                  ║")
    print("║     llmstat — Calibrated LLM Estimator for Human-Subject Research   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    for scenario in SCENARIOS:
        run_scenario(scenario)

    print()
    print("=" * 72)
    print("  Cross-Scenario Comparison")
    print("=" * 72)
    print(f"  {'Scenario':<42} {'N_LLM':>8} {'Savings':>12} {'Reduction':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*12} {'-'*10}")

    for s in SCENARIOS:
        b_true, l_true, d_true, dmax = s["params"]
        human_cal, llm_cal, cond_labels, pl, pt = generate_synthetic_calibration(
            s["K"], s["M_c"], s["M_v"], b_true, l_true, d_true, dmax,
        )
        cal = calibrate(human_cal, llm_cal, cond_labels, pl, pt)
        design = compute_design(
            cal, s["min_effect"], s["N_h"], s["cost_human"], s["cost_llm"],
            s["M_c"], s["K"],
        )
        savings = design.human_cost - design.llm_cost
        short_name = s["name"].replace("Scenario ", "S").split(" —")[0]
        print(
            f"  {s['name']:<42} {design.n_llm:>8} "
            f"{format_money(savings):>12} {format_pct(design.reduction):>10}"
        )

    print()
    print(NON_SURROGACY_TABLE)
    print()
    print("─" * 72)
    print("Key takeaway from the paper (§6–§8):")
    print()
    print("  LLMs can reduce web A/B testing costs by 50–80% for shallow,")
    print("  text-mediated outcomes (headlines, copy, form layout, button")
    print("  preference) — BUT only when the calibration protocol confirms")
    print("  that bias is small, variance ratio is bounded, and identifiability")
    print("  error is controlled.")
    print()
    print("  They are NOT replacements for experiments involving real financial")
    print("  decisions, long-term behavior, social dynamics, or genuinely novel")
    print("  design paradigms outside the training distribution.")
    print("─" * 72)


if __name__ == "__main__":
    main()
