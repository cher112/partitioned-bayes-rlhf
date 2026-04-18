# Partitioned Bayes Error for RLHF Preference Data

> **Code and data supplementary** for a Master's research proposal (UTokyo / Sugiyama-Ishida lab, 2026). This repository hosts preliminary experiments that motivate a 2-year research program on **calibration-aware Bayes error estimation** for RLHF preference data. The proposal itself is a separate document; this README contains the full empirical backing.

---

## TL;DR

- Ishida (ICLR 2023 oral) gives an instance-free estimator `R̂* = (1/N) Σ min(c_i, 1-c_i)` for the Bayes error from soft labels.
- In RLHF, the "soft labels" come from LLM judges, which are **systematically miscalibrated**. Plugging in naively yields estimates dominated by **classifier capacity**, not data-side noise.
- We establish this empirically on a 2-Gaussian toy (**23× spread in R̂* at fixed accuracy**, Fig. 1) and on 5 open-source 7B/8B LLM judges on HelpSteer2 (**Pearson `r(accuracy, R̂*_iso) = -0.98`**, Fig. 4).
- HelpSteer2's 9125 human-annotator preferences show genuine **3.3× data-side heterogeneity** across partition strength bins (Fig. 2) — classifier-independent.
- A first-pass corrected estimator (isotonic + plug-in) pins `R̂*` to the true value within ±0.001 across 10 temperatures on the toy (Fig. 3), proving the correction is feasible in the monotone regime. The Year-1 research target is the **rank-breaking multi-model** regime.

---

## Four headline figures

### Fig. 1 — Plug-in R̂* is driven by calibration, not data noise

![P0 V3](experiments/P0-synthetic-confounding/fig_synthetic_rstar_v3.png)

Single trained MLP on a 2-Gaussian problem with known `R* = Φ(-1) ≈ 0.1587`. Scanning softmax temperature `T ∈ {0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0}` changes nothing about the classifier (accuracy held at 0.836) but makes plug-in `R̂*` span 0.017 → 0.387 (**23×**). The signed calibration bias perfectly predicts the `R̂*` gap (Pearson `r = -1.00`, p ≈ 10⁻⁵³). Unsigned ECE loses the sign and correlates at only r = +0.44. Script: [synthetic_demo_v3.py](experiments/P0-synthetic-confounding/synthetic_demo_v3.py). Raw data: [results_v3.json](experiments/P0-synthetic-confounding/results_v3.json).

### Fig. 2 — HelpSteer2 human-annotator partition (no classifier involved)

![P1](experiments/P1-human-partition/fig_partition_hs2pref.png)

HelpSteer2-Preference 9125 pairs, each scored by 3 annotators with `strength ∈ {0, ±1, ±2, ±3}`. Partitioning by `|aggregate strength|`:

| Partition `|strength|` | n_pairs | Agreement | Per-annotator error (vs majority) |
|---|---|---|---|
| 0 (tie) | 2007 | 0.671 | **0.331** |
| 1 (slight) | 3033 | 0.902 | 0.121 |
| 2 (moderate) | 2562 | 0.876 | 0.154 |
| 3 (strong) | 1523 | 0.917 | **0.099** |

3.3× range in per-annotator error rate, established without any classifier or softmax. Script: [analyze_preference.py](experiments/P1-human-partition/analyze_preference.py). Raw data: [partition_stats.json](experiments/P1-human-partition/partition_stats.json).

### Fig. 3 — Corrected estimator works (toy, single-model regime)

![P0 V4](experiments/P0-synthetic-confounding/fig_synthetic_rstar_v4.png)

Four estimators on the same 10-temperature grid:

| Estimator | span across T | mean | gap vs 0.159 |
|---|---|---|---|
| plug-in raw | **0.371** | 0.165 | depends heavily on T |
| 1 − accuracy | 0.000 | 0.161 | flat (argmax invariant) |
| **isotonic-calibrated R̂\*** | **0.001** | 0.157 | pinned to true value |
| margin-matching R̂\*_CA | 0.001 | 0.157 | pinned to true value |

Isotonic calibration's rank-based nature makes it invariant to temperature scaling. Under this monotone regime, the corrected estimator achieves the target consistency. Script: [synthetic_demo_v4.py](experiments/P0-synthetic-confounding/synthetic_demo_v4.py). Raw data: [results_v4.json](experiments/P0-synthetic-confounding/results_v4.json).

### Fig. 4 — Isotonic alone is insufficient in the multi-model regime (negative result)

![P2](experiments/P2-ece-vs-rstar/fig_acc_vs_rstar.png)

Five open-source LLM judges (Qwen, Mistral, Granite, OLMo, Falcon) score 1000 HelpSteer2 pairs each with AB/BA debiasing. Per-judge isotonic calibration reduces ECE from 0.03–0.21 to 0.03–0.04, **yet** `R̂*_iso` remains tightly correlated with judge accuracy:

| Judge | accuracy | R̂*_iso | ECE_iso |
|---|---|---|---|
| Granite-3.0-8B | 0.587 | **0.398** | 0.039 |
| Qwen-2.5-7B | 0.583 | 0.414 | 0.043 |
| Mistral-7B-v0.3 | 0.578 | 0.414 | 0.032 |
| OLMo-7B-hf | 0.518 | 0.461 | 0.029 |
| Falcon-7B | 0.519 | **0.471** | 0.030 |

Pearson `r(accuracy, R̂*_iso) = −0.98`, p = 0.003. Since the underlying `(X, Y)` distribution is identical across judges, true `R*` is a single number; the 0.073 span reflects **rank-breaking** calibration differences that single-model isotonic cannot reconcile. Script: [plot.py](experiments/P2-ece-vs-rstar/plot.py). Raw data: [cross_partition_stats.json](experiments/analysis_hs2_5judges/cross_partition_stats.json).

---

## Repository layout

```
partitioned-bayes-rlhf/
├── README.md                                  (this file)
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── build_pairs.py                         HelpSteer2 pair construction
│   ├── calibration_utils.py                   isotonic / temperature / ECE helpers
│   ├── llm_judge_infer.py                     vLLM pairwise preference inference (AB/BA)
│   ├── analyze_partitioned_rstar.py           cross-judge R*_plug / R*_iso + permutation test
│   └── verify_params.py                       tokenizer / prompt-length sanity check
├── run_pipeline.sh                            orchestrates Steps 1–3
├── run_all_judges.sh                          5 judges × single-GPU parallel launcher
├── download_models.sh                         fetch 5 judges from HuggingFace (via HF-Mirror)
└── experiments/
    ├── P0-synthetic-confounding/              2-Gaussian toy, V1–V4
    ├── P1-human-partition/                    HelpSteer2-Preference strength analysis
    ├── P2-ece-vs-rstar/                       5-judge cross-correlation plots
    └── analysis_hs2_5judges/                  raw outputs of 5-judge pipeline
```

`data/` and `models/` are intentionally gitignored — download via `download_models.sh` and HuggingFace datasets (HelpSteer2) respectively. All figures and JSON metrics are versioned.

---

## Setup

### Hardware
Tested on a single 96 GB RTX PRO 6000 (Blackwell) for 7B BF16 inference. Each judge fits in ~20 GB + KV cache. 5 judges in parallel benefit from 5 GPUs; serial execution works with one GPU (~30 minutes total for N = 1000 pairs).

### Software
```bash
conda create -n pbrhf python=3.12
conda activate pbrhf
pip install -r requirements.txt
```

See `requirements.txt` for exact pinned versions. Critical: `vllm==0.19.0`, `torch==2.10.0`. Known issue — two assertion patches in `torch._inductor` are needed for vLLM 0.19 + torch 2.10 combo; see [Issues](#known-issues) below.

### Data
HelpSteer2 rating subset loads from HuggingFace automatically on first run. For P1 (preference subset):

```bash
export HF_ENDPOINT=https://hf-mirror.com    # or omit if outside China
huggingface-cli download nvidia/HelpSteer2 --repo-type dataset \
    --local-dir data/hs2_extra \
    --include 'preference/preference.jsonl.gz' 'disagreements/disagreements.jsonl.gz'
```

### Models
```bash
bash download_models.sh     # ~85 GB total across 5 models
```

---

## Reproducing the four figures

### Fig. 1 (P0 V3) — temperature-scaling confound

```bash
cd experiments/P0-synthetic-confounding
python synthetic_demo_v3.py    # ~30 s on single GPU
```

Output: `fig_synthetic_rstar_v3.png`, `results_v3.json`. Random seed: 42.

### Fig. 2 (P1) — HelpSteer2 human partition

```bash
cd experiments/P1-human-partition
python analyze_preference.py    # ~10 s, CPU only
```

Requires `data/hs2_extra/preference/preference.jsonl.gz` (see Setup).

### Fig. 3 (P0 V4) — corrected estimator proof-of-life

```bash
cd experiments/P0-synthetic-confounding
python synthetic_demo_v4.py    # ~30 s on single GPU
```

### Fig. 4 (P2) — 5-judge correlation

Prerequisite: complete the 5-judge inference pipeline first.

```bash
# Full pipeline: build pairs → 5-judge inference → cross-judge analysis
bash run_pipeline.sh 1000

# Then the scatter plot:
cd experiments/P2-ece-vs-rstar
python plot.py
```

`run_pipeline.sh` takes ~30 minutes on 5 GPUs in parallel (or ~2.5 hours serial on one GPU).

---

## Full results tables

### P0 V3 — raw data

| T | acc | avg_conf | signed_bias | ECE | sharpness | R̂* | gap vs 0.159 |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.836 | 0.983 | +0.148 | 0.111 | 0.484 | 0.017 | −0.142 |
| 0.20 | 0.836 | 0.967 | +0.132 | 0.128 | 0.467 | 0.033 | −0.126 |
| 0.30 | 0.836 | 0.951 | +0.116 | 0.115 | 0.451 | 0.049 | −0.110 |
| 0.50 | 0.836 | 0.918 | +0.083 | 0.083 | 0.418 | 0.082 | −0.077 |
| 0.75 | 0.836 | 0.879 | +0.044 | 0.044 | 0.379 | 0.121 | −0.038 |
| 1.00 | 0.836 | 0.843 | +0.008 | 0.009 | 0.343 | 0.157 | −0.002 |
| 1.50 | 0.836 | 0.783 | −0.052 | 0.052 | 0.283 | 0.217 | +0.058 |
| 2.00 | 0.836 | 0.738 | −0.098 | 0.098 | 0.238 | 0.262 | +0.104 |
| 3.00 | 0.836 | 0.676 | −0.160 | 0.160 | 0.176 | 0.324 | +0.165 |
| 5.00 | 0.836 | 0.613 | −0.222 | 0.222 | 0.113 | 0.387 | +0.228 |

### P0 V4 — stability of corrected estimators

| Estimator | Span across T | Mean across T |
|---|---|---|
| plug-in R̂*_raw | 0.371 | 0.165 |
| 1 − accuracy | 0.000 | 0.161 |
| isotonic R̂*_iso | 0.001 | 0.157 |
| margin-CA R̂*_CA | 0.001 | 0.157 |

### P1 — per-partition statistics

| Partition | n_pairs | Agreement | Err rate | Strength Var |
|---|---|---|---|---|
| 0 | 2007 | 0.6714 | 0.3312 | 0.038 |
| 1 | 3033 | 0.9017 | 0.1206 | 0.172 |
| 2 | 2562 | 0.8763 | 0.1539 | 0.401 |
| 3 | 1523 | 0.9173 | 0.0993 | 0.217 |

### P2 — 5-judge full metrics

| Judge | n | acc vs gold | R*_raw | R*_iso | ECE_raw | ECE_iso |
|---|---|---|---|---|---|---|
| Falcon-7B-Instruct | 999 | 0.5185 | 0.4699 | 0.4705 | 0.0279 | 0.0300 |
| Granite-3.0-8B-Instruct | 999 | 0.5866 | 0.2438 | 0.3980 | 0.1800 | 0.0393 |
| Mistral-7B-Instruct-v0.3 | 999 | 0.5776 | 0.2947 | 0.4141 | 0.1368 | 0.0315 |
| OLMo-7B-Instruct-hf | 999 | 0.5175 | 0.3325 | 0.4607 | 0.1540 | 0.0292 |
| Qwen2.5-7B-Instruct | 999 | 0.5826 | 0.2187 | 0.4140 | 0.2120 | 0.0427 |

Cross-judge statistics on `R*_iso`: mean 0.4315, std 0.0287, var 8.21e-4, bootstrap 95 % CI for var [1.0e-5, 2.7e-4], permutation p-value 0.000.

---

## Known issues

### vLLM 0.19 + torch 2.10 compatibility

On recent torch builds the inductor backend crashes at model load with two `AssertionError`s. Current workaround is a small source-level patch:

```bash
# In-place patch 1: torch/_inductor/select_algorithm.py line ~1695
# Replace: assert name not in self.all_templates, "duplicate template name"
# With:    if name in self.all_templates: return

# In-place patch 2: same file line ~2164
# Replace: assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"
# With:    if hasattr(extern_kernels, name): self.name = name; return
```

Plus one missing-attribute fix for Falcon in `vllm/model_executor/models/falcon.py:171`:

```python
# Before:
rope_parameters=config.rope_parameters,
# After:
rope_parameters=getattr(config, "rope_parameters",
                         {"rope_theta": getattr(config, "rope_theta", 10000.0)}),
```

Always pass `enforce_eager=True` to `LLM(...)` to bypass the torch.compile path.

### Tokenizer A/B multi-ID

Every judge encodes "A"/"B" into multiple token IDs depending on leading whitespace. See `get_ab_token_ids()` in `src/llm_judge_infer.py`. Mis-handling drops ~50 % of probability mass in some models.

### OLMo model variant

Use **`allenai/OLMo-7B-Instruct-hf`** (HuggingFace-native architecture), not `allenai/OLMo-7B-Instruct` which has the legacy `OLMoForCausalLM` architecture not supported by vLLM. Also note OLMo's `max_position_embeddings = 2048`; the script sets `max_model_len=2048` for OLMo and Falcon.

---

## Citing

If you build on this code, cite the three foundation papers:

```bibtex
@inproceedings{ishida2023good,
  title={Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification},
  author={Ishida, Takashi and Yamane, Ikko and Charoenphakdee, Nontawat and Niu, Gang and Sugiyama, Masashi},
  booktitle={ICLR},
  year={2023}
}

@article{wang2024helpsteer2,
  title={HelpSteer2: Open-source dataset for training top-performing reward models},
  author={Wang, Zhilin and others},
  journal={arXiv:2406.08673},
  year={2024}
}

@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and others},
  booktitle={SOSP},
  year={2023}
}
```

---

## License

MIT (code). Figures and numerical results are CC-BY-4.0 — attribution to this repository when used in derivative work.
