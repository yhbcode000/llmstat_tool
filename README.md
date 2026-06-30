# LLMStat — Calibrated LLM Estimator for Human-Subject Research

A Python implementation of the calibrated LLM estimator methodology described in *"Pretrained Large Language Models as Statistical Instruments: Restricted Risk Equivalence under Squared Loss"*.

## What the theory actually says — and what it does not

**Short answer: yes — but only in a specific, well-defined sense.**

The paper's framework establishes:

> An LLM is a misspecified estimator of conditional expectations under squared loss.

So it can be used as a statistical tool when:

- the task is **prediction of a measurable outcome**
- the target is a **conditional mean or smooth functional**
- the loss is approximately **squared error or Bregman-type**

### When LLMs ARE valid statistical tools

| Use case | Why it works |
|----------|-------------|
| Survey response prediction | LLM estimates conditional mean of bounded responses |
| Average treatment effect approximation (under strong assumptions) | Linear contrast of conditional means |
| Conditional mean estimation | Direct target of the framework |
| Missing-value imputation | Functional estimator of E[Y\|X] |

This matches the core abstraction: `T(P) = E[Y | X]`.

LLMs reduce cost when:

- human sampling is expensive
- variance dominates measurement error
- calibration bias is controlled

So they behave like: **a cheap Monte Carlo surrogate for conditional expectations.**

From the calibration protocol: if bias `b̂` is small, variance ratio `λ̂` is bounded, and identifiability error `δ` is controlled, then LLMs can substitute human samples with corrected scaling.

### When LLMs are NOT valid statistical tools

| Limitation | Why it fails |
|------------|-------------|
| Causal inference without structure | LLMs do not identify counterfactual effects, structural parameters, or intervention mechanisms |
| Novel experimental regimes | If P(new condition) ∉ support of training data, KL projection is uninformative and ε_rep dominates |
| Mechanism discovery | LLMs approximate outcomes, not generative processes — cannot replace cognitive models, causal graphs, or mechanistic experiments |

### Cost reduction: YES, but only in a specific regime

The framework implies: Total Risk = Irreducible variance + ε_rep² + o(1).

Cost reduction is valid if:

- ε_rep is small or calibrated
- δ is controlled
- the model class is sufficiently rich

**Interpretation:** LLMs act like a **variance-reducing surrogate estimator with fixed bias floor**. They reduce cost by replacing repeated human sampling with high-volume synthetic sampling.

### The key insight

> LLMs are NOT replacements for experiments. They are **low-cost estimators of conditional expectations in a fixed statistical model class.**

That is much narrower — but also mathematically correct.

### Practical takeaway

**Safely use LLMs as:**

- survey simulators (with calibration)
- expectation estimators
- bootstrap-like augmentation tools
- approximate risk estimators

**Do NOT use them as:**

- substitutes for causal experiments
- ground-truth human behavior generators
- validation sources for novel phenomena

### One-line answer

> Yes — LLMs can be used as statistical tools to reduce cost, but only as calibrated misspecified estimators of conditional expectations under squared-loss regimes, not as general substitutes for experimental or causal inference.

## Quick Start

```bash
# Install
pip install -e .

# Copy environment config
cp .env.example .env

# Run a two-condition framing survey simulation
llmstat run --conditions 2 --calibration 500 --output results/
```

## Architecture

- **`llmstat.connector`** — LLM API layer: Docker localai (default) → OpenAI (fallback)
- **`llmstat.methodology`** — Core calibration/estimation pipeline
- **`llmstat.validation`** — TOST equivalence testing
- **`llmstat.report`** — Report generation (table + detailed flows)

## LLM Backend

Two backends are supported, selected via `LLMSTAT_BACKEND` in `.env`:

| Backend | Default | Requires |
|---------|---------|----------|
| `localai` | ✅ | Docker |
| `openai` | fallback | API key |

### LocalAI (Docker)

```bash
# Start localai (CPU)
docker compose --profile cpu up -d

# Pull a model
docker compose exec localai-cpu local-ai run llama-3.2-3b-instruct
```

### OpenAI (fallback)

Set `LLMSTAT_BACKEND=openai` and `OPENAI_API_KEY` in `.env`.

## Usage

```bash
# Calibrate and estimate for a two-condition study
llmstat run \
  --conditions "neutral,treatment" \
  --calibration 500 \
  --equivalence-margin 0.02 \
  --min-effect 0.05

# Estimate calibration parameters from existing data
llmstat calibrate --human-data human.csv --llm-data llm.csv

# Run TOST validation
llmstat validate --human-estimate 0.76 --llm-estimate 0.78 --margin 0.02
```

---

## Examples

Run the web A/B testing demonstration (no API keys needed, uses synthetic data):

```bash
python examples/web_ab_test.py
```

This runs three web A/B testing scenarios through the full calibration pipeline:
headline CTR tests, landing page conversion, and button color click-through.
See `examples/web_ab_test.py` for the full annotated source.

## Citation

```bibtex
@article{yang_llmstat,
  title   = {Pretrained Large Language Models as Statistical Instruments:
             Restricted Risk Equivalence under Squared Loss},
  author  = {Yang, Haobo},
  year    = {2025},
  note    = {Preprint. This repository is the companion implementation.}
}

@article{yang2026transformer,
  title   = {Transformer Architectures as Complete {Bayes} Processes:
             A Formal Proof in the Measure-Theoretic Kernel Framework},
  author  = {Yang, Haobo},
  year    = {2026},
  eprint  = {2606.30440},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  note    = {Cites and uses this repository as companion tooling.}
}
```
