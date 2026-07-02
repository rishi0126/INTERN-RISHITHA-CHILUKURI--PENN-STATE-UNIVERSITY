# Agentic Claims Anomaly Triage System — Proof of Concept

**Cotiviti Intern Assessment — Topic 2: Clinical Decision Making and Pattern Recognition**
**Rishitha Chilukuri**

## What this is

A three-layer pipeline that demonstrates explainable, agentic AI for healthcare payment integrity:

1. **Clustering engine** — groups synthetic providers into behavioral baselines by specialty (K-Means style, per-specialty centroid clustering)
2. **Anomaly detection** — scores each provider's distance from its specialty baseline to flag statistical outliers
3. **Agentic AI layer** — an LLM agent reasons step-by-step through each flagged provider and recommends approve, escalate, or deny, with a plain-language explanation

This matches the architecture proposed in the accompanying written report.

## Files in this submission

- `pipeline.py` — standalone Python implementation of the clustering and anomaly detection layers (stages 1-2 of the pipeline). Run this to see the core data science logic in isolation and validate that planted anomalies are correctly caught.
- `cotiviti_poc_anomaly_triage.html` — the full interactive demo, including all three pipeline layers. Open directly in any browser, no installation required. The agentic AI layer in this file calls the Claude API (Claude Sonnet 4.6) live to generate real-time reasoning for each flagged provider. Human reviewer decisions feed back into the system as a retraining signal, continuously improving cluster baselines over time — see the feedback loop in the architecture diagram and `apply_feedback()` in `pipeline.py`.
- `pipeline_output.json` — example output from running `pipeline.py`, used to validate pipeline correctness.

## How to run

### Option 1: Full interactive demo (recommended for review)
Open `cotiviti_poc_anomaly_triage.html` directly in any web browser. Click "Run pipeline," then click any flagged provider to see the agentic AI layer reason through it live.

### Option 2: Standalone pipeline logic
```bash
python pipeline.py
```
This regenerates the synthetic dataset, runs clustering and anomaly detection, and prints the top flagged providers along with a validation check confirming all planted anomalies were correctly caught.

## Validation

The dataset includes 84 synthetic provider profiles across 5 specialties (Family Medicine, Cardiology, Orthopedics, Dermatology, Physical Therapy), with 4 deliberately planted anomalies representing realistic fraud/error patterns:

- **Upcoding** — billing significantly above peer cost average
- **Phantom billing volume** — claim volume far exceeding plausible capacity
- **Extreme cost outlier** — a single claim type billed at many multiples of peer average
- **Procedure count outlier** — procedures-per-claim inconsistent with specialty norms

Running the pipeline correctly surfaces all 4 planted anomalies at the top of the flagged list, validating that the clustering and anomaly scoring approach works as intended before the agentic AI layer is engaged.

## Design notes

- Clustering is performed **within each specialty**, not globally — comparing a cardiologist's billing pattern to a dermatologist's would be meaningless. This mirrors how a real payment integrity system would never cross-compare fundamentally different practice types.
- The agentic AI layer is intentionally the final stage, not the first. Cheap, fast statistical methods (clustering, distance scoring) do the bulk filtering; the more expensive LLM reasoning step is reserved only for the small number of cases that actually need investigation. This is a deliberate cost and latency optimization that would matter at production scale.
- If the live API call to Claude fails (e.g. network restrictions in the review environment), the HTML demo falls back to a local reasoning renderer that still demonstrates the full three-step explanation structure using the same computed statistics, so the demo never breaks.
