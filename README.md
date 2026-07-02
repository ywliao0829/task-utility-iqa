# Task-Utility IQA

This repository contains code, processed result tables, and visualization scripts for the paper:

**Task-Utility IQA: Rethinking Image Quality Assessment for Embodied Settings**

## Overview

This project studies image quality assessment (IQA) from a task-utility perspective. Instead of evaluating image quality only by task-agnostic perceptual fidelity, we evaluate whether an IQA or utility score can predict downstream VLM task success in embodied settings.

The current release focuses on the VisDrone-300 evidence study used in the paper.

## Repository Structure

```text
configs/              Configuration files and model plans.
src/api_eval/          API-based VLM evaluation and task-success alignment scripts.
src/data/              Dataset construction utilities for VisDrone-style VQA samples.
src/metrics/           Traditional IQA metric computation.
src/analysis/          Baseline scoring, task-utility benchmarking, and analysis utilities.
reproduce/figure/      Main figure reproduction script and processed input table.
results/               Processed result tables and lightweight prediction summaries.
figures/               Generated paper figures.
scripts/               Convenience entry points.
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## API Configuration

Some scripts use API-based VLM inference. Create a local `.env` file from the template:

```bash
cp .env.example .env
```

Then fill in your own API keys. Do not commit real API keys.

## Running API-based VLM Evaluation

The API evaluation scripts are under:

```text
src/api_eval/
```

Example:

```bash
python -m src.api_eval.run_all_strict300_from_plan
```

The model plans used in the paper are provided under:

```text
configs/api_plans/
```

## Computing Task-Utility Alignment

Representative commands:

```bash
python -m src.api_eval.recompute_api_correctness
python -m src.api_eval.compute_all_strict300_alignment
python -m src.api_eval.export_cross_model_compare300_excel
```

These scripts compute answer-normalized task-success labels and aggregate alignment metrics such as AUROC, average precision, and Spearman correlation.

## Baseline and Metric Analysis

Baseline scoring and task-utility benchmarking utilities are provided under:

```text
src/analysis/
src/metrics/
```

These scripts support traditional IQA metrics, MLLM-based IQA baselines, and task-utility alignment comparisons.

## Reproducing the Main Figure

The main paper figure can be reproduced from the processed results:

```bash
cd reproduce/figure
python make_task_utility_composite.py
```

Alternatively, run the convenience entry point from the repository root:

```bash
python scripts/make_task_utility_composite.py
```

The generated figures are also provided under:

```text
figures/
```

## Evaluation Protocol

For each VisDrone image-question pair, we record whether a target VLM answer matches the gold task outcome after answer normalization, yielding a binary task-success label.

Each IQA or utility score is then evaluated as a scalar predictor of downstream task success. We report task-utility alignment using AUROC, average precision, and Spearman correlation.

## Data Note

This repository does not redistribute the full VisDrone dataset. Please download the original VisDrone dataset from the official source.

We provide processed metadata, model plans, result tables, and lightweight prediction summaries needed to reproduce the task-utility alignment analysis reported in the paper.

## Released Materials

This repository includes:

- API-based VLM evaluation scripts;
- VisDrone-style VQA sample construction utilities;
- traditional IQA metric computation scripts;
- task-success and task-utility alignment analysis scripts;
- baseline scoring and comparison utilities;
- processed result tables used in the paper;
- visualization scripts for the main evidence figure;
- generated figures in PNG and PDF formats.

## Citation

The citation entry will be updated after the review process.

## License

This repository is released for academic research purposes.
