# To Memorize or to Retrieve: Scaling the Interaction Between Pretraining and Retrieval

[Karan Singh](https://karanps.com), [Michael Yu](https://michaelc-yu.github.io/), [Varun Gangal](https://vgtomahawk.github.io), Zhuofu Tao, [Sachin Kumar](https://sites.google.com/view/sachinkumar?pli=1), [Emmy Liu](https://nightingal3.github.io), [Steven Y. Feng](https://styfeng.github.io)

**Stanford University, Patronus AI, The Ohio State University, Carnegie Mellon University, and DegenAI Labs**

<hr>

![DCLM corpora ranging from 30M to 100B tokens are used to train OLMo-2 models from 30M to 3B parameters, paired with held-out or reused retrieval stores containing up to 20B tokens. We fit a scaling law over model size, pretraining exposure, and datastore size to compare performance along controlled fixed-budget allocation slices. The example plots illustrate that retrieval’s interaction with scale depends on the evaluation target: gold-answer perplexity gains are front-loaded and larger for smaller models, while decision-accuracy gains favor larger models on the multiple-choice tasks studied.](figures/intro.png)

**Figure: **Overview of the controlled scaling study.** _Left_: We jointly vary OLMo-2 model size, DCLM pretraining exposure, datastore size, and whether the datastore contains held-out or previously seen data. _Center_: The fitted scaling surface characterizes performance across pretraining–retrieval allocations; fixed-token slices provide performance-side comparisons. _Right_: Retrieval gains depend on the evaluation target: gold-answer perplexity gains are front-loaded and largest for smaller models, whereas decision-accuracy gains favor larger models on the multiple-choice benchmarks studied.

Preprint: [arXiv](https://arxiv.org/pdf/2604.00715)

## Abstract

Retrieval-augmented generation (RAG) improves language model (LM) performance by providing relevant context at test time for knowledge-intensive situations. In this work, we systematically study the trade-off between pretraining and retrieval by training OLMo-2-based LMs ranging from 30M to 3B parameters on up to 100B DCLM tokens, while varying pretraining data scale, retrieval store size, and retrieval store source (pretraining vs. new data) across reasoning, scientific QA, and open-domain QA benchmarks. We find that retrieval gains depend on model capacity and pretraining exposure and are strongly front-loaded, with a median 91\% of the largest observed improvement realized by one retrieval token per model parameter. However, the interaction is objective-dependent: smaller models gain more in gold-answer perplexity, whereas larger, more-pretrained models gain more in accuracy. Retrieval from previously seen data also preserves most of the held-out retrieval gain. Retrieval is therefore a task-, regime-, and metric-dependent complement to parametric learning whose value also depends on datastore size and information novelty. Overall, this motivates the explicit partitioning of data between internalization and external access for LM design.

## Overview

This repository contains utilities for:
- preparing split datasets for pretraining-vs-retrieval experiments
- launching pretraining and evaluation jobs
- fitting and analyzing scaling-law trends from aggregated CSV metrics

## Repository Layout

- `data/scripts/`: data download, split, and optimization utilities
- `data/splits/`: canonical shard split CSVs for pretraining/retrieval
- `scripts/eval/`: evaluation orchestration and result aggregation
- `scripts/rag/`: RAG index build scripts and FAISS docs
- `scripts/pretraining/`: pretraining scheduler scripts (`.sbatch`)
- `scripts/litgpt_to_hf/`: LitGPT checkpoint conversion utilities
- `scripts/fit_scaling_law/`: scaling-law fitting + diagnostics

## Getting Started

### 1) Environment

Use Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you use Weights & Biases:

```bash
wandb login
```

### 2) Initialize submodules

```bash
git submodule update --init --recursive
```

### 3) Configure paths via environment variables

```bash
export MODELS_ROOT="/path/to/pretrained/models"
export INDICES_ROOT="/path/to/rag/indices"
export LITGPT_DIR="/path/to/litgpt"
export HF_HOME="/path/to/hf-cache"
```

---

## Workflows

### Dataset split and optimization

See `data/DATA_README.md` and the scripts in `data/scripts/`.

End-to-end helper:

```bash
bash data/scripts/create_datasets.sh
```

### Pretraining

Pretraining job templates are in `scripts/pretraining/`.

Example (submit a 30M pretraining sweep):

```bash
sbatch scripts/pretraining/olmo2_30m.sbatch
```

You can similarly launch other scales:
- `scripts/pretraining/olmo2_136m.sbatch`
- `scripts/pretraining/olmo2_136m_seed43.sbatch`
- `scripts/pretraining/olmo2_233m.sbatch`
- `scripts/pretraining/olmo2_728m.sbatch`
- `scripts/pretraining/olmo2_1b.sbatch`
- `scripts/pretraining/olmo2_3b.sbatch`

### RAG index generation

RAG index build utilities are in `scripts/rag/`.

Primary index build script:

```bash
bash scripts/rag/build_ratioed_indices.sh
```

Supporting files:
- `scripts/rag/build_ratioed_indices.py`
- `scripts/rag/FAISS_ON_GPU.md`

### Multi-index RAG evaluation

Main runner:

```bash
bash scripts/eval/run_all_evals.sh
```

This script:
- prepares eval task inputs
- generates retrieval JSONL files for each index/task
- runs baseline and RAG evaluation for each model scale

### Scaling-Law Fitting

```bash
python3 scripts/fit_scaling_law/fit_scaling_law.py \
  --dir /path/to/aggregated_csvs \
  --metric "perplexity,none" \
  --mode sequential \
  --retrieval_model log
```

Related analysis:
- Sigma/kappa analysis: `scripts/fit_scaling_law/detect_saturation.py`
- Robustness/stability to random seeds: `scripts/fit_scaling_law/seed_stability_study.py`

## Citation

If you found our work helpful for your own research or applications, please cite it using the following BibTeX:
```bibtex
@misc{RAGScalingLaws2026,
      title={To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining}, 
      author={Karan Singh and Michael Yu and Varun Gangal and Zhuofu Tao and Sachin Kumar and Emmy Liu and Steven Y. Feng},
      year={2026},
      eprint={2604.00715},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.00715}, 
}
```
