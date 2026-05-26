# Task-Utility IQA

This repository contains the evidence-study code, processed result tables, and visualization scripts for the paper:

**Task-Utility IQA: Rethinking Image Quality Assessment for Embodied Settings**

## Overview

Image quality assessment (IQA) is commonly evaluated by task-agnostic perceptual fidelity. However, in embodied settings, the key question is not only whether an image looks good, but whether it preserves the visual evidence needed for a downstream decision.

This repository provides the code and processed results for our VisDrone-300 evidence study, where each IQA or utility score is evaluated by its ability to predict downstream VLM task success.

The current release includes:

- processed cross-model result tables,
- task-utility alignment metrics,
- scripts for generating the main evidence figure,
- generated figures in PNG and PDF formats.

## Repository Structure

```text
scripts/    Analysis and visualization scripts.
results/    Processed result tables used in the paper.
figures/    Generated paper figures.
data/       Metadata or processed sample lists for the VisDrone-300 subset.
```

## Main Figure Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the visualization script:

```bash
python scripts/make_task_utility_composite.py
```

The generated figures will be saved to the output directory specified in the script.

## Data Note

This repository does not redistribute the full VisDrone dataset. Please download the original dataset from the official VisDrone source. The processed metadata and result tables provided here are used only for reproducing the reported task-utility alignment analysis.

## Released Materials

The repository includes:

- `scripts/make_task_utility_composite.py`: script for generating the main task-utility evidence figure;
- `results/`: processed evaluation results and metric summaries;
- `figures/`: generated figures used in the paper;
- `data/`: metadata or sample lists for the VisDrone-300 evidence study.

## License

This repository is released for academic research purposes.
