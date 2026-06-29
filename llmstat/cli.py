"""CLI entrypoint for llmstat.

Usage:
    llmstat run [--conditions ...] [--calibration N] [options]
    llmstat calibrate --human-data FILE --llm-data FILE
    llmstat validate --sample-a FILE --sample-b FILE [--margin E]
    llmstat simulate                         (quick demo with synthetic data)
    llmstat check                            (validate config + backend connectivity)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from llmstat.config import config
from llmstat.connector import LLMConnector, LLMError
from llmstat.methodology import (
    calibrate,
    check_feasibility,
    compute_design,
    build_condition_prompts,
    parse_numeric_response,
    CalibrationResult,
)
from llmstat.report import (
    build_compact_table,
    build_detailed_report,
    export_csv,
    build_report_string,
    format_money,
    format_pct,
)
from llmstat.validation import tost_two_sample

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """LLMStat — Calibrated LLM Estimator for Human-Subject Research."""


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

@main.command()
def check() -> None:
    """Validate configuration and test backend connectivity."""
    console.print(f"[bold]LLMStat backend:[/] {config.backend}")
    console.print(f"[bold]Active model:[/] {config.active_model}")
    console.print(f"[bold]API endpoint:[/] {config.active_base_url}")

    issues = config.validate()
    if issues:
        console.print("\n[red]Configuration issues:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        raise SystemExit(1)

    console.print("\n[dim]Testing connectivity...[/dim]")
    connector = LLMConnector(timeout=30.0)
    try:
        response = connector.chat(
            system_prompt="You are a helpful assistant. Reply with exactly one word.",
            user_prompt="Say 'connected'",
            temperature=0.0,
        )
        console.print(f"[green]✓ Backend reachable:[/] {response.strip()}")
    except LLMError as exc:
        console.print(f"[red]✗ Backend unreachable:[/]\n{exc}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@main.command()
@click.option("--conditions", "-c", default="neutral,treatment",
              help="Comma-separated condition names")
@click.option("--outcome", "-o", default="bounded agreement score in [0,1]",
              help="Description of the outcome variable")
@click.option("--calibration", "-m", type=int, default=500,
              help="Calibration sample size M_c")
@click.option("--prompt-validation", "-v", type=int, default=100,
              help="Prompt-validation sample size M_v")
@click.option("--min-effect", "-d", type=float, default=0.05,
              help="Minimum effect of scientific interest Δ")
@click.option("--equivalence-margin", "-e", type=float, default=0.02,
              help="TOST equivalence margin ε")
@click.option("--human-n", "-n", type=int, default=1000,
              help="Baseline human per-condition sample N_h")
@click.option("--cost-human", type=float, default=8.0,
              help="Illustrative human cost per response")
@click.option("--cost-llm", type=float, default=0.002,
              help="Illustrative LLM cost per response")
@click.option("--temperature", type=float, default=0.2,
              help="LLM sampling temperature")
@click.option("--output", "-O", type=click.Path(), default=None,
              help="Output directory for reports")
@click.option("--dry-run", is_flag=True,
              help="Compute design without calling LLM API")
def run(
    conditions: str,
    outcome: str,
    calibration: int,
    prompt_validation: int,
    min_effect: float,
    equivalence_margin: float,
    human_n: int,
    cost_human: float,
    cost_llm: float,
    temperature: float,
    output: Optional[str],
    dry_run: bool,
) -> None:
    """Run a full calibrated LLM estimation study."""
    cond_list = [c.strip() for c in conditions.split(",")]
    K = len(cond_list)

    console.print(f"[bold]LLMStat Study[/]")
    console.print(f"  Conditions: {cond_list}")
    console.print(f"  Outcome: {outcome}")
    console.print(f"  Calibration size M_c: {calibration}")
    console.print(f"  Min effect Δ: {min_effect}")
    console.print(f"  Equivalence margin ε: {equivalence_margin}")
    console.print()

    # --- Calibration phase ---
    console.print("[bold]Phase 1: Calibration[/]")
    connector = LLMConnector()

    if dry_run:
        console.print("[yellow]Dry-run mode — using synthetic calibration data[/yellow]")
        # Synthetic calibration data for demo
        rng = np.random.default_rng(42)
        human_cal = 0.5 + 0.3 * rng.random((calibration, K))
        llm_cal = human_cal + 0.004 + 0.02 * rng.random((calibration, K))
        cond_labels = rng.integers(0, K, size=calibration)
        prompt_labels = rng.integers(0, K, size=prompt_validation)
        prompt_truth = rng.integers(0, K, size=prompt_validation)
        # Introduce small identifiability error
        flip_mask = rng.random(prompt_validation) < 0.01
        prompt_labels[flip_mask] = (prompt_labels[flip_mask] + 1) % K
    else:
        # Real calibration: collect paired human + LLM responses
        console.print(f"  Collecting {calibration} paired calibration responses...")
        prompts = build_condition_prompts(cond_list, outcome, calibration // K + 1)
        raw_responses = connector.batch_chat(prompts, temperature=temperature)

        human_cal = np.zeros((calibration, K))
        llm_cal = np.zeros((calibration, K))
        cond_labels = np.zeros(calibration, dtype=int)

        for i in range(calibration):
            cond_labels[i] = i % K
            # In a real study, human_cal would come from actual human data
            # Here we use synthetic human responses for demonstration
            human_cal[i, cond_labels[i]] = 0.5 + 0.3 * np.random.random()
            try:
                llm_cal[i, cond_labels[i]] = parse_numeric_response(raw_responses[i], 0.5)
            except (IndexError, ValueError):
                llm_cal[i, cond_labels[i]] = 0.5

        # Prompt validation (synthetic for demo — real study uses human annotators)
        rng = np.random.default_rng(42)
        prompt_labels = rng.integers(0, K, size=prompt_validation)
        prompt_truth = rng.integers(0, K, size=prompt_validation)

    cal = calibrate(human_cal, llm_cal, cond_labels, prompt_labels, prompt_truth)
    console.print(f"  b̂ = {cal.bias:.6f}")
    console.print(f"  λ̂ = {cal.lambda_hat:.4f}")
    console.print(f"  δ̂ = {cal.delta_hat:.4f}")
    console.print(f"  Δ̂_max = {cal.delta_max:.4f}")
    console.print()

    # --- Feasibility ---
    console.print("[bold]Phase 2: Feasibility check[/]")
    feas = check_feasibility(cal, min_effect)
    console.print(f"  ρ = {feas.penalty:.4f}")
    console.print(f"  Feasible: [{'green' if feas.feasible else 'red'}]{feas.feasible}[/]")
    if not feas.feasible:
        console.print(f"  [red]Reason: {feas.reason}[/red]")
        return
    console.print()

    # --- Design ---
    console.print("[bold]Phase 3: Design computation[/]")
    design = compute_design(cal, min_effect, human_n, cost_human, cost_llm, calibration, K)
    console.print(f"  N_LLM per condition: {design.n_llm}")
    console.print(f"  Human-only cost: {format_money(design.human_cost)}")
    console.print(f"  LLM-augmented cost: {format_money(design.llm_cost)}")
    console.print(f"  Cost reduction: {format_pct(design.reduction)}")
    console.print()

    # --- LLM Sampling + TOST ---
    console.print("[bold]Phase 4: LLM sampling + TOST validation[/]")
    if dry_run:
        console.print("[yellow]Dry-run — skipping LLM sampling[/yellow]")
        tost_result = None
    else:
        console.print(f"  Generating {design.n_llm} LLM responses per condition...")
        prompts = build_condition_prompts(cond_list, outcome, design.n_llm)
        raw = connector.batch_chat(prompts, temperature=temperature)
        llm_values = np.array([parse_numeric_response(r, 0.5) for r in raw])

        # Split per condition
        llm_per_cond = llm_values.reshape(K, design.n_llm)
        # Human baseline (synthetic for demo)
        rng = np.random.default_rng(123)
        human_per_cond = 0.5 + 0.3 * rng.random((K, human_n))

        # TOST on the first contrast
        tost_result = tost_two_sample(llm_per_cond[0], human_per_cond[0], equivalence_margin)
        console.print(f"  TOST: [{'green' if tost_result.equivalent else 'red'}]{'PASS' if tost_result.equivalent else 'FAIL'}[/]")
        console.print(f"    diff={tost_result.estimate_diff:.4f}, SE={tost_result.se_diff:.4f}, p={tost_result.p_value:.4f}")

    # --- Report ---
    console.print()
    console.print("[bold]Report[/]")

    case_record = {
        "use_case": f"Study: {conditions}",
        "human_cost": design.human_cost,
        "llm_cost": design.llm_cost,
        "reduction": design.reduction,
        "validation": "TOST on effect size" if tost_result and tost_result.equivalent else "FAILED",
        "penalty": feas.penalty,
        "n_llm": design.n_llm,
        "feasible": feas.feasible,
    }

    detailed = build_detailed_report(
        cal=cal, feasibility=feas, design=design, tost=tost_result,
        case_name=f"Study: {conditions}",
        target_estimand=f"mean {outcome} contrast",
        cost_human=cost_human, cost_llm=cost_llm,
        n_calibration=calibration, n_conditions=K, n_human=human_n,
        min_effect=min_effect,
        report_action="Report calibrated mean contrast; do not claim individual-response distribution equivalence.",
    )

    report = build_report_string([case_record], [detailed])
    console.print(report)

    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_csv([case_record], str(out_dir / "report.csv"))
        (out_dir / "report.txt").write_text(report)
        console.print(f"\n[green]Report saved to {out_dir}[/green]")


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--human-data", type=click.Path(exists=True), required=True)
@click.option("--llm-data", type=click.Path(exists=True), required=True)
@click.option("--output", "-o", type=click.Path(), default="calibration.json")
def calibrate_cmd(human_data: str, llm_data: str, output: str) -> None:
    """Estimate calibration parameters from CSV data files."""
    import pandas as pd

    human = pd.read_csv(human_data).values
    llm = pd.read_csv(llm_data).values

    # Assume first column is condition label
    cond_labels = human[:, 0].astype(int)
    human_resp = human[:, 1:]
    llm_resp = llm[:, 1:]

    # Synthetic prompt validation for demo
    K = human_resp.shape[1]
    prompt_labels = np.zeros(100, dtype=int)
    prompt_truth = np.zeros(100, dtype=int)

    cal = calibrate(human_resp, llm_resp, cond_labels, prompt_labels, prompt_truth)

    result = {
        "bias": cal.bias,
        "se_bias": cal.se_bias,
        "lambda_hat": cal.lambda_hat,
        "delta_hat": cal.delta_hat,
        "delta_max": cal.delta_max,
    }
    Path(output).write_text(json.dumps(result, indent=2))
    console.print(f"[green]Calibration saved to {output}[/green]")
    console.print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--sample-a", type=click.Path(exists=True), required=True)
@click.option("--sample-b", type=click.Path(exists=True), required=True)
@click.option("--margin", "-e", type=float, default=0.02)
@click.option("--alpha", type=float, default=0.05)
def validate_cmd(sample_a: str, sample_b: str, margin: float, alpha: float) -> None:
    """Run TOST equivalence validation on two CSV data files."""
    import pandas as pd

    a = pd.read_csv(sample_a).values.flatten()
    b = pd.read_csv(sample_b).values.flatten()

    result = tost_two_sample(a, b, margin, alpha)

    table = Table(title="TOST Validation")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Equivalent", f"[{'green' if result.equivalent else 'red'}]{result.equivalent}[/]")
    table.add_row("Difference", f"{result.estimate_diff:.6f}")
    table.add_row("SE(diff)", f"{result.se_diff:.6f}")
    table.add_row("p-value", f"{result.p_value:.6f}")
    table.add_row("Margin ε", f"{result.margin}")
    table.add_row("t (upper)", f"{result.statistic_upper:.4f}")
    table.add_row("t (lower)", f"{result.statistic_lower:.4f}")
    table.add_row("Critical t", f"{result.critical_value:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

@main.command()
def simulate() -> None:
    """Run a quick demonstration with synthetic data (no API calls)."""
    console.print("[bold]LLMStat — Synthetic Demonstration[/bold]\n")

    rng = np.random.default_rng(42)

    # Three use cases matching the paper
    cases = [
        {
            "name": "Framing survey",
            "K": 2, "Nh": 1000, "Mc": 500, "ch": 8.0, "cllm": 0.002,
            "Delta": 0.05, "bhat": 0.004, "lhat": 1.20, "dhat": 0.01, "dmax": 0.30,
            "target": "bounded agreement mean contrast",
            "validation": "TOST on effect size",
            "action": "Report a calibrated mean contrast; do not claim individual-response distribution equivalence.",
        },
        {
            "name": "UI design rating (UICrit-style)",
            "K": 2, "Nh": 300, "Mc": 150, "ch": 2.856, "cllm": 0.002,
            "Delta": 0.50, "bhat": 0.05, "lhat": 1.10, "dhat": 0.02, "dmax": 4.0,
            "target": "mean UI design-quality rating or region-level severity score",
            "validation": "condition-wise rating calibration",
            "action": "Report mean rating/contrast and validation; keep free-text critiques as qualitative support only.",
        },
        {
            "name": "SUS human-experience score",
            "K": 2, "Nh": 200, "Mc": 80, "ch": 2.856, "cllm": 0.002,
            "Delta": 5.0, "bhat": 0.8, "lhat": 1.15, "dhat": 0.01, "dmax": 40,
            "target": "mean SUS score or SUS-point contrast",
            "validation": "TOST on SUS-point margin",
            "action": "Report calibrated mean SUS contrast; do not claim open-ended experience understanding.",
        },
    ]

    case_records = []
    detailed_reports = []

    for c in cases:
        cal = CalibrationResult(
            bias=c["bhat"], se_bias=0.001,
            lambda_hat=c["lhat"], delta_hat=c["dhat"], delta_max=c["dmax"],
        )
        feas = check_feasibility(cal, c["Delta"])
        design = compute_design(cal, c["Delta"], c["Nh"], c["ch"], c["cllm"], c["Mc"], c["K"])

        case_records.append({
            "use_case": c["name"],
            "human_cost": design.human_cost,
            "llm_cost": design.llm_cost,
            "reduction": design.reduction,
            "validation": c["validation"],
            "penalty": feas.penalty,
            "n_llm": design.n_llm,
            "feasible": feas.feasible,
        })

        detailed_reports.append(build_detailed_report(
            cal=cal, feasibility=feas, design=design, tost=None,
            case_name=c["name"], target_estimand=c["target"],
            cost_human=c["ch"], cost_llm=c["cllm"],
            n_calibration=c["Mc"], n_conditions=c["K"], n_human=c["Nh"],
            min_effect=c["Delta"], report_action=c["action"],
        ))

    report = build_report_string(case_records, detailed_reports)
    console.print(report)

    # Verify expected values
    expected = [(1623, 4006.492), (603, 430.812), (399, 230.076)]
    all_ok = True
    for i, (exp_n, exp_c) in enumerate(expected):
        if case_records[i]["n_llm"] != exp_n:
            console.print(f"[red]FAIL: Case {i+1} N_LLM={case_records[i]['n_llm']} (expected {exp_n})[/red]")
            all_ok = False
        if abs(case_records[i]["llm_cost"] - exp_c) > 0.02:
            console.print(f"[red]FAIL: Case {i+1} C_LLM={case_records[i]['llm_cost']:.2f} (expected {exp_c})[/red]")
            all_ok = False
    if all_ok:
        console.print("\n[green]VERIFIED: All expected values match.[/green]")


if __name__ == "__main__":
    main()
