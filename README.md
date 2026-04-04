# Multi-Modal Generative AI: IP-Adapter Image Conditioning and Audio Generation/Understanding

### A Systematic Parametric Study of IP-Adapter Image Variation, Block-Level Style Transfer, Automatic Speech Recognition, and Text-to-Speech Synthesis Using Stable Diffusion XL, Whisper, and Bark

---

![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Diffusers: 0.25+](https://img.shields.io/badge/%F0%9F%A4%97_Diffusers-0.25+-dfb317)
![Transformers: 4.36+](https://img.shields.io/badge/%F0%9F%A4%97_Transformers-4.36+-dfb317)
![Platform](https://img.shields.io/badge/Platform-Google_Colab-F9AB00?logo=googlecolab&logoColor=white)

---

**Author:** Ryan Kamp  
**Affiliation:** Department of Computer Science, University of Cincinnati  
**Location:** Cincinnati, OH, USA  
**Email:** kamprj@mail.uc.edu  
**GitHub:** https://github.com/ryanjosephkamp  
**Course:** CS6078 Generative AI  
**Assignment:** Assignment 13 — IP-Adapter Image Conditioning & Audio Generation/Understanding  
**Date:** April 2026

---

## Table of Contents

1. [Abstract](#abstract)
2. [Project Overview](#project-overview)
3. [Research Questions](#research-questions)
4. [The Five Scripts](#the-five-scripts)
5. [Experimental Design](#experimental-design)
6. [Key Findings](#key-findings)
7. [Selected Results](#selected-results)
8. [Getting Started](#getting-started)
9. [Usage](#usage)
10. [Project Structure](#project-structure)
11. [Reports and Documentation](#reports-and-documentation)
12. [References](#references)
13. [License](#license)

---

## Abstract

This project presents a systematic parametric study spanning two generative AI modalities — **vision** and **audio** — based on Chapters 8 and 9 of *Hands-on Generative AI with Transformers and Diffusion Models* (Alammar & Grootendorst, O'Reilly, 2024). Through seven experiments encompassing **149 generated images**, **29 synthesized audio files**, **164 ASR transcriptions**, and **339 total parameter configurations**, the study investigates:

- **IP-Adapter uniform-scale image variation** via SDXL Base with decoupled cross-attention (GA18B)
- **Block-level style transfer and content–style disentanglement** via per-layer IP-Adapter scale control (GA18C)
- **Text–image interaction** under simultaneous IP-Adapter and text prompt conditioning (GA18B + GA18C)
- **Automatic speech recognition** with Whisper model comparison and sampling rate robustness analysis (GA19B)
- **Text-to-speech synthesis** with Bark, evaluating generation consistency and round-trip fidelity (GA19C)

All experiments were executed on Google Colab with NVIDIA A100/H100 GPUs using FP16 precision, accumulating **828.2 seconds** of total GPU computation time. Key findings include: a recommended IP-Adapter scale range of $\lambda \in [0.4, 0.7]$ for balancing reference fidelity and creative variation; effective content–style disentanglement through selective upsampling-block conditioning; Whisper-medium achieving a word error rate (WER) of 0.172 on conversational banking-domain speech; and Bark TTS generating speech at approximately $2.3\times$ real-time with high seed-dependent variability.

---

## Project Overview

### Course Context

This project extends a homework assignment for **CS6078 Generative AI** (Spring 2026, University of Cincinnati). The assignment is drawn from Chapters 8 (Image Prompting with IP-Adapter) and 9 (Audio Generation and Understanding) of the course textbook.

### Original Assignment Description

> *"Run GA18B.py and GA18C.py for image prompting and also GA19B.py for automatic speech recognition and show your results. You may change the images and other parameters."*

### Research Extension

Rather than merely running the three required scripts at default settings, this project designs and executes a comprehensive experimental study that:

1. **Reproduces baseline results** from all five scripts (GA18B, GA18C, GA19A, GA19B, GA19C) with their default parameters.
2. **Systematically varies key parameters** — IP-Adapter scales, block-level configurations, reference images, text prompts, ASR models, sampling rates, TTS inputs, and random seeds — to map the controllable generation design space across both modalities.
3. **Addresses seven research questions** concerning parameter sensitivity, content–style disentanglement, text–image interaction, ASR robustness, TTS consistency, seed sensitivity, and cross-modal comparison.
4. **Produces publication-quality deliverables** — a comprehensive markdown report, an IEEE-formatted LaTeX report, and professional repository documentation.

### Technical Foundation

The project spans two modalities and two distinct generative paradigms:

**Vision (Chapter 8) — IP-Adapter Image Conditioning.** The IP-Adapter introduces a **decoupled cross-attention** mechanism into the Stable Diffusion XL U-Net, enabling image-conditioned generation without retraining the base model. A CLIP image encoder extracts a reference image embedding, which is injected through dedicated key/value projections operating in parallel with the existing text cross-attention layers:

$$\mathbf{Z} = \text{Attention}(\mathbf{Q}, \mathbf{K}_t, \mathbf{V}_t) + \lambda \cdot \text{Attention}(\mathbf{Q}, \mathbf{K}_i, \mathbf{V}_i)$$

where $\lambda$ is the IP-Adapter scale controlling image conditioning strength, $\mathbf{K}_t, \mathbf{V}_t$ are text-conditioned keys/values, and $\mathbf{K}_i, \mathbf{V}_i$ are image-conditioned keys/values. GA18B applies a uniform $\lambda$ across all layers; GA18C uses a **per-block scale dictionary** to selectively condition specific U-Net upsampling layers, enabling content–style disentanglement.

**Audio (Chapter 9) — ASR and TTS.** GA19B demonstrates automatic speech recognition using the Hugging Face `transformers` pipeline API with models such as wav2vec 2.0 and OpenAI Whisper, applied to the MINDS-14 banking-domain speech dataset. GA19C demonstrates text-to-speech synthesis using the Bark model, a multi-stage autoregressive architecture that converts text to semantic tokens, coarse acoustic tokens, fine acoustic tokens, and finally raw audio waveforms via neural codec decoding.

---

## Research Questions

This study addresses seven research questions spanning both vision and audio modalities:

| RQ | Question | Modality | Experiment |
|----|----------|----------|------------|
| **RQ1** | How does the uniform IP-Adapter scale $\lambda$ govern the trade-off between reference image fidelity and output variation? | Vision | Exp 2 |
| **RQ2** | How do different block-level IP-Adapter scale configurations affect the type and degree of style transfer in content–style disentanglement? | Vision | Exp 3 |
| **RQ3** | How do IP-Adapter image conditioning and text prompt conditioning interact when both are active simultaneously? | Vision | Exp 4 |
| **RQ4** | How do different ASR models compare on conversational banking-domain speech, and how robust is ASR to sampling rate mismatch? | Audio | Exp 5 |
| **RQ5** | How consistent is Bark's TTS output across repeated generations, and how do text input properties affect synthesized speech quality? | Audio | Exp 6 |
| **RQ6** | How sensitive are IP-Adapter image variations and Bark TTS outputs to the random seed? | Cross-modal | Exp 7 |
| **RQ7** | How do generative control mechanisms differ between vision (adapter-based diffusion) and audio (autoregressive TTS, pipeline-based ASR)? | Cross-modal | Cross-analysis |

<details>
<summary><strong>Expanded Research Question Descriptions</strong></summary>

**RQ1 — IP-Adapter Uniform Scale Sensitivity.** The IP-Adapter scale $\lambda$ controls the relative influence of the image conditioning signal via the decoupled cross-attention mechanism: $\mathbf{Z} = \mathbf{Z}_{\text{text}} + \lambda \cdot \mathbf{Z}_{\text{image}}$. This RQ systematically characterizes the full functional relationship between $\lambda$ and perceptual similarity to the reference image. *Hypothesis:* The fidelity–variation relationship is nonlinear, with the most perceptually interesting output in $\lambda \in [0.4, 0.8]$.

**RQ2 — Block-Level Style Transfer.** GA18C demonstrates content–style disentanglement by applying IP-Adapter conditioning to a single cross-attention layer within `block_0` of the U-Net's upsampling path. The design space of block-level configurations is vast. *Hypothesis:* Different upsampling blocks transfer different stylistic attributes — deeper blocks transfer global style qualities; shallower blocks transfer localized textural details.

**RQ3 — Text–Image Interaction.** GA18B uses an empty text prompt (pure image conditioning); GA18C uses a non-empty prompt with block-level conditioning. This RQ investigates the interaction when uniform IP-Adapter conditioning is combined with an active text prompt. *Hypothesis:* At low $\lambda$ (< 0.3), text dominates; at high $\lambda$ (> 0.7), the image dominates; the intermediate range produces a blend with gradual transition.

**RQ4 — ASR Model Comparison and Robustness.** This RQ has two parts: (a) comparing pre-trained ASR models (wav2vec 2.0, Whisper-tiny, Whisper-small, Whisper-medium) on MINDS-14 banking-domain speech; (b) quantifying the impact of sampling rate mismatch on transcription quality. *Hypothesis:* Whisper models outperform wav2vec 2.0; performance improves with model size; incorrect sampling rate produces significant WER increases.

**RQ5 — TTS Consistency and Variation.** Bark is a stochastic generative model whose autoregressive sampling introduces variation in prosody, intonation, and timing. *Hypothesis:* Repeated generations produce consistent semantic content but varying prosodic realization; shorter texts yield more consistent outputs; complex or domain-specific texts may introduce disfluencies.

**RQ6 — Seed Sensitivity.** For IP-Adapter, the random seed controls the initial noise tensor; for Bark, it controls autoregressive sampling across three transformer stages. *Hypothesis:* IP-Adapter exhibits decreasing seed sensitivity as $\lambda$ increases; Bark exhibits moderate sensitivity with consistent lexical content but varying prosody.

**RQ7 — Cross-Modal Control Comparison.** A structural comparison of continuous latent diffusion with adapter-based conditioning (GA18B/GA18C), discrete token-based autoregressive generation (GA19C), and pipeline-based discriminative inference (GA19B). *Hypothesis:* Diffusion-based generation offers more fine-grained continuous control; TTS provides more precise content specification via text; ASR occupies a distinct position with objective WER/CER metrics.

</details>

---

## The Five Scripts

The project builds on five Python scripts spanning two chapters and two modalities:

| Script | Chapter | Modality | Paradigm | Model / Pipeline | Key Innovation |
|--------|---------|----------|----------|-----------------|----------------|
| **GA18B.py** | 8 | Vision | Image Variation | SDXL Base 1.0 + IP-Adapter (`ip-adapter_sdxl.bin`) | Decoupled cross-attention with uniform $\lambda$ scale |
| **GA18C.py** | 8 | Vision | Style Transfer | SDXL Base 1.0 + IP-Adapter (`ip-adapter_sdxl.bin`) | Block-level per-layer scale for content–style disentanglement |
| **GA19A.py** | 9 | Audio | Data Exploration | Hugging Face `datasets` + Gradio | MINDS-14 dataset inspection and interactive UI |
| **GA19B.py** | 9 | Audio | ASR (Speech → Text) | `transformers` pipeline (wav2vec 2.0 / Whisper) | Pre-trained ASR with audio resampling to 16 kHz |
| **GA19C.py** | 9 | Audio | TTS (Text → Speech) | Bark (`suno/bark-small`) | Multi-stage autoregressive audio generation via EnCodec |

### Vision Progression: GA18B → GA18C

The vision scripts demonstrate **increasing granularity of control** over IP-Adapter image conditioning:

- **GA18B (Uniform Scale):** A single scalar $\lambda \in [0, 1]$ is applied equally across all U-Net cross-attention layers. At $\lambda = 0$, image conditioning is disabled; at $\lambda = 1$, it operates at full strength. With an empty text prompt, generation is driven entirely by the CLIP image embedding of the reference image.
- **GA18C (Block-Level Scale):** A per-block, per-layer scale dictionary (`{"up": {"block_0": [0.0, 1.0, 0.0]}}`) selectively applies image conditioning to individual cross-attention layers within specific upsampling blocks. This enables **content–style disentanglement**: the text prompt controls semantic content (e.g., "a cat inside of a box") while the reference image's style (color palette, texture, artistic character) is injected through targeted upsampling layers. The technique exploits the hierarchical feature encoding property of U-Net architectures — deeper layers encode global structure; shallower upsampling layers encode fine-grained stylistic details.

### Audio Progression: GA19A → GA19B → GA19C

The audio scripts form a complete **bidirectional audio processing pipeline**:

- **GA19A (Data Exploration):** Loads the MINDS-14 banking-domain speech dataset (`PolyAI/minds14`, en-AU subset) and provides a Gradio interface for interactive exploration of audio samples and intent labels.
- **GA19B (Speech → Text):** Runs automatic speech recognition on MINDS-14 audio clips using the Hugging Face `transformers` pipeline API. Audio is resampled to 16 kHz — the standard rate expected by ASR models — ensuring correct temporal interpretation of the signal.
- **GA19C (Text → Speech):** Synthesizes speech from text using Bark's multi-stage autoregressive architecture: text → semantic tokens → coarse acoustic tokens → fine acoustic tokens → raw audio waveform (via EnCodec neural codec decoding).

---

## Experimental Design

### Overview

The study comprises **seven experiments** executed across **three GPU-optimized execution groups** on Google Colab (NVIDIA A100/H100, FP16 precision). Each group loads its primary model(s) once and runs all associated experiments before unloading, minimizing redundant model loading overhead.

| Group | Model(s) | Experiments | Primary Output |
|-------|----------|-------------|----------------|
| **A — Vision** | SDXL Base 1.0 + IP-Adapter | Exp 1 (baselines), 2, 3, 4, 7 (vision seeds) | 149 images |
| **B — ASR** | wav2vec 2.0, Whisper-tiny/small/medium | Exp 1 (baseline), 5 | 164 transcriptions |
| **C — TTS** | Bark (`suno/bark-small`) | Exp 1 (baseline), 6, 7 (TTS seeds) | 29 audio files |

### Experiment Summary

| Exp | Name | Script(s) | RQ | Parameter Space | Outputs |
|-----|------|-----------|-----|----------------|---------|
| **1** | Baseline Reproduction | GA18B, GA18C, GA19A, GA19B, GA19C | — | All defaults, seed=42 | 2 images, 1 transcription, 1 audio |
| **2** | Uniform Scale Sensitivity | GA18B | RQ1 | $\lambda \in \{0.0, 0.1, \ldots, 1.0\}$; 3 seeds; alt. reference image; inference steps ∈ {10, 25, 50} | 51 images |
| **3** | Block-Level Configurations | GA18C | RQ2 | 6 layer configs, 3 upsampling blocks, 5 scale magnitudes, 4 text prompts, 2 reference images | 46 images |
| **4** | Text–Image Interaction | GA18B + GA18C | RQ3 | 6 uniform scales with active text, 3 prompt specificity levels, 3 semantic compatibility levels, block vs. uniform | 34 images |
| **5** | ASR Model Comparison | GA19B | RQ4 | 4 ASR models × 20 samples; 4 sampling rates × 2 models × 10 samples | 164 transcriptions |
| **6** | TTS Consistency | GA19C | RQ5 | 5 seeds (consistency), 3 text lengths, 3 domains, round-trip TTS→ASR | 29 audio files + 3 transcriptions |
| **7** | Seed Sensitivity | GA18B, GA18C, GA19C | RQ6 | 3 paradigms × 8 seeds ({42, 123, 456, 789, 1024, 2048, 3000, 4096}) | 16 images + 8 audio files |

<details>
<summary><strong>Sub-Experiment Breakdown</strong></summary>

**Experiment 2 — Sub-experiments:**
- **2A:** Scale sweep $\lambda \in \{0.0, 0.1, \ldots, 1.0\}$ × 3 seeds → 33 images
- **2B:** 6 scale values × 2 seeds with alternative reference image (`SampleURL.Mamoeiro`) → 12 images
- **2C:** Inference steps ∈ {10, 25, 50} at $\lambda=0.8$ × 2 seeds → 6 images

**Experiment 3 — Sub-experiments:**
- **3A:** 6 layer configurations within `block_0` × 3 seeds → 18 images
- **3B:** 3 upsampling blocks (`block_0`, `block_1`, `block_2`) × 2 seeds → 6 images
- **3C:** Block-level scale $s \in \{0.25, 0.5, 0.75, 1.0, 1.5\}$ × 2 seeds → 10 images
- **3D:** 4 text prompts × 2 seeds → 8 images
- **3E:** 2 reference images × 2 seeds → 4 images

**Experiment 4 — Sub-experiments:**
- **4A:** 6 uniform scales with active text prompt × 2 seeds → 12 images
- **4B:** 3 prompt specificity levels at $\lambda=0.5$ × 2 seeds → 6 images
- **4C:** 3 semantic compatibility levels at $\lambda=0.5$ × 2 seeds → 6 images
- **4D:** 2 prompts with block-level config × 2 seeds → 4 images
- **4E:** Uniform vs. block-level direct comparison × 3 seeds → 6 images

**Experiment 5 — Sub-experiments:**
- **5A:** 4 ASR models × 20 samples → 80 transcriptions
- **5B:** 4 sampling rates × 2 models × 10 samples → 80 transcriptions

**Experiment 6 — Sub-experiments:**
- **6A:** 5 seeds, same text → 5 audio files (consistency baseline)
- **6B:** 3 text lengths × 2 seeds → 6 audio files
- **6C:** 3 domains (conversational, technical, proper nouns) × 2 seeds → 6 audio files
- **6D:** Round-trip TTS→ASR on 3 texts → 3 audio files + 3 transcriptions

</details>

### Computational Budget

| Metric | Value |
|--------|-------|
| Total GPU time | **828.2 seconds** |
| Mean time per image (GA18B) | 1.58 s |
| Mean time per image (GA18C) | 1.44 s |
| Mean time per ASR transcription | 0.28 s |
| Mean time per TTS audio (Bark) | 19.2 s |
| Platform | Google Colab (NVIDIA A100 / H100), FP16 precision |

---

## Key Findings

### Vision — IP-Adapter Image Conditioning

1. **Uniform scale operates as a smooth, nonlinear, monotonic dial** (RQ1). The IP-Adapter scale $\lambda$ produces no abrupt perceptual thresholds; the steepest change in reference fidelity occurs between $\lambda = 0.2$ and $\lambda = 0.5$. The recommended operating range for balancing reference fidelity and creative variation is $\lambda \in [0.4, 0.7]$.

2. **Block-level conditioning enables effective content–style disentanglement** (RQ2). Restricting IP-Adapter conditioning to the deepest upsampling block (`block_0`) transfers global stylistic attributes (color temperature, tonal character) while preserving text-prompt-specified content. Shallower blocks (`block_2`) produce minimal visible style transfer. Multi-layer activation (`[1,1,1]`) produces the strongest style transfer via additive conditioning. The optimal block-level scale range is $s \in [0.5, 1.0]$.

3. **Text and image conditioning blend predictably under uniform scaling** (RQ3). More specific text prompts retain greater influence at moderate $\lambda$; semantically compatible text–image pairs produce more coherent outputs. Uniform scaling blends content and style from the reference image simultaneously, while block-level scaling separates them.

4. **Generation time is consistent across scales.** Mean generation time per image is 1.58 s (GA18B uniform) and 1.44 s (GA18C block-level) on A100/H100 GPUs with FP16 precision.

### Audio — ASR and TTS

5. **Whisper-medium achieves the lowest WER on banking-domain speech** (RQ4). On 20 MINDS-14 (en-AU) samples at 16 kHz: Whisper-medium WER = **0.172**, Whisper-small WER = **0.180**, Whisper-tiny WER = **0.259**, wav2vec 2.0 WER = **0.407**. The marginal gain from Whisper-small to Whisper-medium is modest while compute more than doubles (0.42 s → 0.74 s per sample).

6. **Sampling rate mismatch severely degrades ASR** (RQ4). Providing 8 kHz audio without resampling to the expected 16 kHz increases WER by **+83%** for wav2vec 2.0 (0.407 → 0.747) and **+129%** for Whisper-small (0.180 → 0.412). CER follows a comparable trend (wav2vec 2.0: 0.211 → 0.533; Whisper-small: 0.103 → 0.344).

7. **Bark TTS generates speech at approximately $2.3\times$ real-time** (RQ5). Generation time scales linearly with output duration across text lengths (short: 7.5 s for 3.2 s audio; long: 31.9 s for 13.8 s audio). Mean generation time across all experiments is 19.2 s per audio file.

8. **Bark exhibits high output variability across seeds** (RQ5). For the same input text, audio duration varies by 74% (7.4–12.9 s), and RMS amplitude spans a 6-fold range (0.023–0.143). Round-trip TTS→ASR WER ranges from 0.000 to 0.250 depending on text content.

### Cross-Modal — Seed Sensitivity

9. **Seed sensitivity differs dramatically across paradigms** (RQ6). Generation time coefficient of variation (CoV) across 8 seeds: GA18C = **0.3%** (near-deterministic), GA18B = **2.8%** (low variability), Bark = **17.3%** (high variability, generation times spanning 17.4–30.2 s). The ranking from lowest to highest seed sensitivity is GA18C < GA18B < GA19C.

10. **Control granularity inversely correlates with stochasticity** (RQ7). Image generation paradigms offer 5–6 controllable parameters with low-to-moderate stochasticity; ASR is fully deterministic with 3 parameters; Bark TTS has only 2 controllable parameters but exhibits the highest output variability. Computational cost spans two orders of magnitude: ASR at 0.28 s/output vs. TTS at 19.2 s/output.

---

## Selected Results

### Baseline Reproduction (Experiment 1)

![Baseline gallery showing default outputs from all four generative scripts](hw13_reports/figures/exp1_baseline_gallery.png)

**Figure 1.** Baseline outputs from GA18B (IP-Adapter image variation, $\lambda = 0.8$), GA18C (block-level style transfer), GA19B (ASR transcription), and GA19C (Bark TTS waveform), all generated with default parameters and seed 42.

---

### IP-Adapter Uniform Scale Sweep (Experiment 2)

![Grid of GA18B outputs across IP-Adapter scale values from 0.0 to 1.0](hw13_reports/figures/ga18b_scale_sweep_grid.png)

**Figure 2.** GA18B image variations across the full uniform IP-Adapter scale range $\lambda \in \{0.0, 0.1, \ldots, 1.0\}$. At $\lambda = 0.0$, the output is unconditioned noise; as $\lambda$ increases, the generated image progressively converges toward the reference. The steepest perceptual transition occurs between $\lambda = 0.2$ and $\lambda = 0.5$.

---

### Block-Level Style Transfer (Experiment 3)

![Comparison of block-level IP-Adapter configurations showing content–style disentanglement](hw13_reports/figures/ga18c_block_comparison.png)

**Figure 3.** Block-level IP-Adapter style transfer comparison across upsampling blocks. `block_0` (deepest) transfers global color palette and tonal character from the reference image while the text prompt ("a cat inside of a box") independently controls semantic content. `block_2` (shallowest) produces minimal visible style transfer.

---

### ASR Model Comparison (Experiment 5)

![Bar chart comparing WER across four ASR models on MINDS-14 banking-domain speech](hw13_reports/figures/asr_model_comparison_bar.png)

**Figure 4.** Word Error Rate (WER) comparison of four ASR models on 20 MINDS-14 (en-AU) samples at 16 kHz. Whisper-medium achieves the lowest WER (0.172), followed by Whisper-small (0.180), Whisper-tiny (0.259), and wav2vec 2.0 (0.407). Error bars indicate standard deviation across samples.

---

### Seed Sensitivity Across Paradigms (Experiment 7)

![Summary of seed sensitivity analysis showing CoV differences across GA18B, GA18C, and GA19C](hw13_reports/figures/seed_sensitivity_summary.png)

**Figure 5.** Cross-paradigm seed sensitivity comparison. Generation time coefficient of variation (CoV) across 8 seeds: GA18C block-level conditioning is near-deterministic (CoV = 0.3%), GA18B uniform conditioning shows low variability (CoV = 2.8%), and Bark TTS exhibits substantial variability (CoV = 17.3%), reflecting the compounding stochasticity of its three-stage autoregressive architecture.

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **CUDA-compatible GPU** (NVIDIA A100 or H100 recommended) or **Google Colab** with GPU runtime
- **~20 GB disk space** for model weights (SDXL Base 1.0, IP-Adapter, Whisper, Bark)
- **Git** for repository cloning

### Installation

```bash
# Clone the repository
git clone https://github.com/ryanjosephkamp/kamp_hw13.git
cd kamp_hw13

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

The [`requirements.txt`](requirements.txt) includes all necessary packages: PyTorch, Hugging Face `diffusers`, `transformers`, `datasets`, `accelerate`, `jiwer` (ASR evaluation), `gradio` (data exploration), and scientific computing libraries.

---

## Usage

### Colab Execution Workflow

All experiments were designed for execution on Google Colab with GPU runtimes. The project follows a **two-stage workflow**:

1. **Dry-run on A100:** Debug and validate experiment logic with reduced parameter sets (fewer seeds, fewer scale values). Notebooks in `hw13_scripts/notebooks/` with `_dry_run` suffix.
2. **Full run on H100:** Execute the complete experimental design with all parameter configurations. Notebooks with `_full` suffix.

### Execution Groups

Experiments are organized into three GPU-optimized execution groups. Each group loads its primary model(s) once and runs all associated experiments before unloading.

| Group | Notebook | Model(s) Loaded | Experiments |
|-------|----------|----------------|-------------|
| **A — Vision** | `group_a_full.ipynb` | SDXL Base 1.0 + IP-Adapter | Exp 1 (vision baselines), 2, 3, 4, 7 (vision seeds) |
| **B — ASR** | `group_b_full.ipynb` | wav2vec 2.0, Whisper-tiny/small/medium | Exp 1 (ASR baseline), 5 |
| **C — TTS** | `group_c_full.ipynb` | Bark (`suno/bark-small`) | Exp 1 (TTS baseline), 6, 7 (TTS seeds) |

### Orchestrator Script

The main orchestrator [`hw13_scripts/kamp_hw13.py`](hw13_scripts/kamp_hw13.py) coordinates all experimental runs. It supports:

- **Checkpoint-based resumption** for Colab session resilience (saves progress to `hw13_experiments/checkpoint.json`)
- **Group-selective execution** — run only the experiments for the current GPU group
- **Automatic CSV logging** of all experimental parameters and results
- **Figure generation** via [`hw13_scripts/hw13_figure_gen.py`](hw13_scripts/hw13_figure_gen.py) and cross-analysis via [`hw13_scripts/hw13_cross_analysis.py`](hw13_scripts/hw13_cross_analysis.py)

### Running Individual Scripts

The five original scripts can be run standalone for quick demonstrations:

```bash
# Image variation with IP-Adapter (uniform scale)
python hw13_scripts/GA18B.py

# Style transfer with IP-Adapter (block-level scale)
python hw13_scripts/GA18C.py

# Audio dataset exploration (launches Gradio UI)
python hw13_scripts/GA19A.py

# Automatic speech recognition
python hw13_scripts/GA19B.py

# Text-to-speech synthesis
python hw13_scripts/GA19C.py
```

> **Note:** Standalone execution requires a CUDA-compatible GPU and downloads model weights on first run.

---

## Project Structure

<details>
<summary><strong>Full Directory Tree</strong></summary>

```
kamp_hw13/
├── LICENSE                            # MIT License
├── README.md                          # This file
├── README_print_format.md             # PDF-export-formatted README
├── requirements.txt                   # Python dependencies
│
├── hw13_background/                   # Course background materials
│   ├── DF510.png
│   ├── hw13.png
│   ├── Image Prompting.pdf
│   └── Generating Audio.pdf
│
├── hw13_scripts/                      # Python scripts
│   ├── GA18B.py                       # IP-Adapter image variation (Ch. 8)
│   ├── GA18C.py                       # IP-Adapter style transfer (Ch. 8)
│   ├── GA19A.py                       # Audio dataset exploration (Ch. 9)
│   ├── GA19B.py                       # Automatic speech recognition (Ch. 9)
│   ├── GA19C.py                       # Text-to-speech synthesis (Ch. 9)
│   ├── kamp_hw13.py                   # Main orchestrator script
│   ├── hw13_data_utils.py             # Data loading and caching utilities
│   ├── hw13_experiment_runner.py       # Experiment execution engine
│   ├── hw13_figure_gen.py             # Figure generation for reports
│   ├── hw13_cross_analysis.py         # Cross-modal analysis and comparisons
│   ├── notebooks/                     # Colab execution notebooks
│   │   ├── group_a_dry_run.ipynb      # Vision experiments — A100 debug
│   │   ├── group_a_full.ipynb         # Vision experiments — H100 full
│   │   ├── group_b_dry_run.ipynb      # ASR experiments — A100 debug
│   │   ├── group_b_full.ipynb         # ASR experiments — H100 full
│   │   ├── group_c_dry_run.ipynb      # TTS experiments — A100 debug
│   │   └── group_c_full.ipynb         # TTS experiments — H100 full
│   └── verification_scripts/          # Step verification tests
│       ├── verify_step3.py
│       ├── verify_step4.py
│       └── verify_step7.py
│
├── hw13_experiments/                   # Experiment results
│   ├── checkpoint.json                # Checkpoint for resumption
│   ├── summary_statistics.json        # Aggregated summary statistics
│   ├── cached_inputs/                 # Cached reference images
│   ├── exp1_baselines/                # Experiment 1: Baseline reproduction
│   ├── exp2_ga18b_scale/              # Experiment 2: Uniform scale sweep
│   ├── exp3_ga18c_blocks/             # Experiment 3: Block-level configs
│   ├── exp4_text_image_interaction/   # Experiment 4: Text–image interaction
│   ├── exp5_ga19b_asr/                # Experiment 5: ASR model comparison
│   ├── exp6_ga19c_tts/                # Experiment 6: TTS consistency
│   ├── exp7_seed_sensitivity/         # Experiment 7: Seed sensitivity
│   ├── group_a_results_20260401_211757/  # Group A raw Colab output
│   ├── group_b_dry_run/               # Group B dry-run output (A100 debug)
│   ├── group_b_results_20260401_223737/  # Group B raw Colab output
│   └── group_c_results_20260402_023519/  # Group C raw Colab output
│
├── hw13_printouts/                    # Console logs from Colab execution
│   ├── all_groups_full_log.txt
│   ├── group_a_full_log.txt
│   ├── group_b_dry_run_log.txt
│   ├── group_b_full_log.txt
│   └── group_c_full_log.txt
│
└── hw13_reports/                      # Final deliverables
    ├── _verify_figs.py                # Figure verification utility
    ├── figures/                        # Report figures (29 figures)
    ├── latex/                          # IEEE-formatted LaTeX report
    │   ├── final_report.tex
    │   ├── final_report.pdf            # Compiled PDF
    │   └── figures/
    ├── markdowns/                     # Markdown reports
    │   ├── hw13_comprehensive_report.md  # Comprehensive research report
    │   └── figures/                   # Markdown-embedded figures
    └── pdfs/                          # PDF exports
```

</details>

---

## Reports and Documentation

| Deliverable | Location | Description |
|------------|----------|-------------|
| **Comprehensive Report** | [`hw13_reports/markdowns/hw13_comprehensive_report.md`](hw13_reports/markdowns/hw13_comprehensive_report.md) | Full research report with methodology, results for all seven experiments, discussion, and conclusions |
| **IEEE LaTeX Report** | [`hw13_reports/latex/final_report.tex`](hw13_reports/latex/final_report.tex) | Publication-formatted version of the comprehensive report |
| **Figures** | [`hw13_reports/figures/`](hw13_reports/figures/) | 29 publication-quality figures including scale sweeps, block comparisons, ASR bar charts, TTS waveforms, and seed sensitivity analyses |
| **Experiment CSVs** | [`hw13_experiments/exp*/`](hw13_experiments/) | Per-experiment CSV files logging all parameter configurations, timing data, and output metadata |
| **Console Logs** | [`hw13_printouts/`](hw13_printouts/) | Complete console output from all three Colab execution groups |
| **Summary Statistics** | [`hw13_experiments/summary_statistics.json`](hw13_experiments/summary_statistics.json) | Aggregated quantitative summary across all experiments |

---

## References

[1] J. Alammar and M. Grootendorst, *Hands-on Generative AI with Transformers and Diffusion Models.* O'Reilly Media, 2024.

[2] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models," *arXiv:2308.06721*, 2023.

[3] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller, J. Penna, and R. Rombach, "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis," *arXiv:2307.01952*, 2023.

[4] D. Gerz, I. Vulic, E. M. Ponti, J. Buber, N. Mrksic, S. Coope, A. Razavi, M. Steedman, and M. Henderson, "Multilingual and Cross-Lingual Intent Detection from Spoken Data," *EMNLP*, pp. 4698–4713, 2021.

[5] Suno AI, "Bark: Text-Prompted Generative Audio Model," GitHub repository, 2023.

[6] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, vol. 33, pp. 6840–6851, 2020.

[7] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR*, pp. 10684–10695, 2022.

[8] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, pp. 234–241, 2015.

[9] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, pp. 8748–8763, 2021.

[10] J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," *arXiv:2207.12598*, 2022.

[11] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," *NeurIPS*, vol. 33, pp. 12449–12460, 2020.

[12] A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," *ICML*, pp. 369–376, 2006.

[13] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," *ICML*, pp. 28492–28518, 2023.

[14] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, "High Fidelity Neural Audio Compression," *TMLR*, 2023.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
