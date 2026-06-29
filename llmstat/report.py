"""Report generation — outputs the paper-style methodology report.

Produces:
  - Compact cost-comparison table (matching Table 2 in the paper)
  - Detailed case-by-case report-generation guide
  - CSV export of computed values
"""

from __future__ import annotations

import csv
import math
from io import StringIO

from llmstat.methodology import CalibrationResult, DesignResult, FeasibilityResult
from llmstat.validation import TOSTResult


def format_money(value: float) -> str:
    """Format a dollar amount."""
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    """Format as percentage with 2 decimal places."""
    return f"{value * 100:.2f}%"


def build_compact_table(
    cases: list[dict],
) -> str:
    """Build the compact cost-comparison table string.

    Parameters
    ----------
    cases : list of dict with keys:
        use_case, human_cost, llm_cost, reduction, validation
    """
    lines = [
        "Use case               Human-only cost   LLM-augmented cost   Reduction   Validation target",
        "-" * 105,
    ]
    for c in cases:
        lines.append(
            f"{c['use_case']:<24s}"
            f"{format_money(c['human_cost']):>18s}"
            f"{format_money(c['llm_cost']):>20s}"
            f"{format_pct(c['reduction']):>12s}"
            f"   {c['validation']}"
        )
    return "\n".join(lines)


def build_detailed_report(
    cal: CalibrationResult,
    feasibility: FeasibilityResult,
    design: DesignResult,
    tost: TOSTResult | None,
    case_name: str,
    target_estimand: str,
    cost_human: float,
    cost_llm: float,
    n_calibration: int,
    n_conditions: int,
    n_human: int,
    min_effect: float,
    report_action: str,
) -> str:
    """Build a detailed case-by-case report block."""
    lines = [
        f"\n{'=' * 70}",
        f"  Case: {case_name}",
        f"{'=' * 70}",
        f"  Target estimand: {target_estimand}",
        f"  Human baseline: K={n_conditions}, N_h={n_human}/condition, cost={format_money(cost_human)}/response",
        f"  Calibration: M_c={n_calibration}, b̂={cal.bias:.4f}, λ̂={cal.lambda_hat:.2f}, δ̂={cal.delta_hat:.4f}, Δ̂_max={cal.delta_max:.4f}",
        f"  Feasibility penalty: ρ={feasibility.penalty:.4f} (<1 required); feasible={feasibility.feasible}",
    ]
    if feasibility.feasible:
        lines.extend([
            f"  Adjusted LLM sample size: N_LLM={design.n_llm} per condition",
            f"  Cost: human-only={format_money(design.human_cost)}, LLM-augmented={format_money(design.llm_cost)}, reduction={format_pct(design.reduction)}",
        ])
    else:
        lines.append(f"  DESIGN INFEASIBLE: {feasibility.reason}")

    if tost is not None:
        lines.extend([
            f"  TOST: {'PASS' if tost.equivalent else 'FAIL'}",
            f"    diff={tost.estimate_diff:.4f}, SE={tost.se_diff:.4f}, p={tost.p_value:.4f}, margin={tost.margin}",
        ])

    lines.extend([
        f"  Paper report guidance: {report_action}",
        f"  If feasibility or validation fails: reject the LLM estimate and report failure, not a substantive effect.",
    ])
    return "\n".join(lines)


def export_csv(cases: list[dict], path: str) -> None:
    """Export case results to CSV."""
    fieldnames = [
        "use_case", "human_cost", "llm_cost", "reduction",
        "validation", "penalty", "n_llm", "feasible",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)


def build_report_string(
    cases: list[dict],
    detailed_reports: list[str],
    uicrit_summary: str | None = None,
) -> str:
    """Combine all report components into a single printable string."""
    parts = [
        "=" * 70,
        "  LLMStat — Applied Methodology Report",
        "=" * 70,
    ]
    if uicrit_summary:
        parts.append(f"\n{uicrit_summary}")
    parts.append("\n--- Compact table values ---")
    parts.append(build_compact_table(cases))
    parts.append("\n--- Detailed report-generation flows ---")
    parts.extend(detailed_reports)
    return "\n".join(parts)
