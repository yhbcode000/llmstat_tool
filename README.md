# LLMStat — Calibrated LLM Estimator for Human-Subject Research

A Python implementation of the calibrated LLM estimator methodology described in *"Pretrained Large Language Models as Statistical Instruments: Restricted Risk Equivalence under Squared Loss"*.

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
