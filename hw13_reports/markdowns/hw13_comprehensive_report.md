# Multi-Modal Generative AI with IP-Adapter and Neural Audio Models: A Systematic Experimental Study

### A Comprehensive Experimental Study of Uniform and Block-Level Image Conditioning, Text-Image Interaction, Automatic Speech Recognition Robustness, and Bark Text-to-Speech Variation

---

**Author:** Ryan Kamp  
**Affiliation:** Department of Computer Science, University of Cincinnati  
**Location:** Cincinnati, OH, USA  
**Email:** kamprj@mail.uc.edu  
**GitHub:** https://github.com/ryanjosephkamp

**Course:** CS6078 Generative AI  
**Assignment:** Homework 13 - IP-Adapter Image Conditioning, ASR, and TTS  
**Date:** March 2026

---

## Abstract

This project presents a systematic parametric study spanning two generative AI modalities — **vision** and **audio** — based on Chapters 8 and 9 of *Hands-on Generative AI with Transformers and Diffusion Models* (Alammar & Grootendorst, O'Reilly, 2024). Through seven experiments encompassing **149 generated images**, **29 synthesized audio files**, **164 ASR transcriptions**, and **339 total parameter configurations**, the study investigates:

- **IP-Adapter uniform-scale image variation** via SDXL Base with decoupled cross-attention (GA18B)
- **Block-level style transfer and content–style disentanglement** via per-layer IP-Adapter scale control (GA18C)
- **Text–image interaction** under simultaneous IP-Adapter and text prompt conditioning (GA18B + GA18C)
- **Automatic speech recognition** with Whisper model comparison and sampling rate robustness analysis (GA19B)
- **Text-to-speech synthesis** with Bark, evaluating generation consistency and round-trip fidelity (GA19C)

All experiments were executed on Google Colab with NVIDIA A100/H100 GPUs using FP16 precision, accumulating **828.2 seconds** of total GPU computation time. Key findings include: a recommended IP-Adapter scale range of $\lambda \in [0.4, 0.7]$ for balancing reference fidelity and creative variation; effective content–style disentanglement through selective upsampling-block conditioning; Whisper-medium achieving a word error rate (WER) of 0.172 on conversational banking-domain speech; and Bark TTS generating speech at approximately $2.3\times$ real-time with high seed-dependent variability.

**Keywords:** IP-Adapter, Stable Diffusion XL, latent diffusion models, content-style disentanglement, automatic speech recognition, text-to-speech synthesis, Bark, wav2vec 2.0, Whisper, multi-modal generative AI

---

## Table of Contents

1. [Introduction](#i-introduction)
2. [Literature Review](#ii-literature-review)
3. [Methodology](#iii-methodology)
4. [Results: Experiment 1 - Baseline Reproduction](#iv-results-experiment-1---baseline-reproduction)
5. [Results: Experiment 2 - IP-Adapter Uniform Scale Sensitivity (GA18B)](#v-results-experiment-2---ip-adapter-uniform-scale-sensitivity-ga18b)
6. [Results: Experiment 3 - Block-Level Scale Configurations for Content-Style Disentanglement (GA18C)](#vi-results-experiment-3---block-level-scale-configurations-for-content-style-disentanglement-ga18c)
7. [Results: Experiment 4 - Text-Image Conditioning Interaction (GA18B + GA18C)](#vii-results-experiment-4---text-image-conditioning-interaction-ga18b--ga18c)
8. [Results: Experiment 5 - ASR Model Comparison and Sampling-Rate Robustness (GA19B)](#viii-results-experiment-5---asr-model-comparison-and-sampling-rate-robustness-ga19b)
9. [Results: Experiment 6 - Bark TTS Consistency and Variation (GA19C)](#ix-results-experiment-6---bark-tts-consistency-and-variation-ga19c)
10. [Results: Experiment 7 - Seed Sensitivity and Reproducibility](#x-results-experiment-7---seed-sensitivity-and-reproducibility)
11. [Discussion](#xi-discussion)
12. [Conclusion](#xii-conclusion)
13. [Future Work](#xiii-future-work)
14. [References](#xiv-references)

---

## I. Introduction

### A. Assignment Context and Project Scope

This study extends Homework 13 for CS6078 Generative AI (Spring 2026, University of Cincinnati), which is based on Chapters 8 and 9 of *Hands-on Generative AI with Transformers and Diffusion Models* by Alammar and Grootendorst [1]. The base assignment requires running three scripts — GA18B.py and GA18C.py for IP-Adapter image prompting and GA19B.py for automatic speech recognition — and presenting the results. This project extends the assignment into a rigorous multi-modal research study by: (1) running all five scripts (GA18B, GA18C, GA19A, GA19B, GA19C) with default parameters to establish baselines, (2) systematically varying parameters across seven experiments to address seven research questions spanning vision and audio modalities, and (3) producing publication-quality analysis with quantitative metrics, comparative grids, and cross-modal synthesis.

The vision domain experiments use the IP-Adapter framework [2] with Stable Diffusion XL (SDXL) [3] to investigate uniform-scale image variation, block-level content-style disentanglement, and text-image conditioning interaction. The audio domain experiments evaluate multiple ASR models on the MINDS-14 banking-domain speech dataset [4], analyze Bark text-to-speech synthesis [5] consistency and variation, and quantify the effects of preprocessing choices on model performance. A cross-modal analysis synthesizes findings across all paradigms.

<div style="page-break-after: always;"></div>

### B. Research Questions

#### RQ1. Uniform IP-Adapter Scale and the Reference-Fidelity vs. Variation Trade-Off

How does the uniform IP-Adapter scale $\lambda$ govern the trade-off between reference image fidelity and output variation in image-prompted generation?

#### RQ2. Block-Level IP-Adapter Configurations and Content-Style Disentanglement

How do different block-level IP-Adapter scale configurations affect the type and degree of style transfer in content-style disentanglement?

#### RQ3. Text-Image Conditioning Interaction Under Joint Prompt and Reference Guidance

How do the IP-Adapter image conditioning and text prompt conditioning interact when both are active simultaneously?

#### RQ4. ASR Model Accuracy and Sensitivity to Sampling-Rate Mismatch

How do different ASR models compare in transcription accuracy on conversational banking-domain speech, and how robust is ASR performance to sampling rate mismatch?

#### RQ5. Bark TTS Consistency, Variation, and Input-Text Effects

How consistent is Bark's TTS output across repeated generations, and how do text input properties (length, complexity, domain) affect the quality and character of the synthesized speech?

#### RQ6. Seed Sensitivity Across Image Generation and Text-to-Speech Synthesis

How sensitive are IP-Adapter image variations and Bark TTS outputs to the random seed, and do some configurations exhibit more deterministic behavior than others?

#### RQ7. Cross-Modal Trade-Offs Between Vision and Audio Control Mechanisms

How do the generative control mechanisms differ between the vision domain (IP-Adapter image conditioning via diffusion) and the audio domain (autoregressive token-based TTS, pipeline-based ASR), and what are their relative strengths and trade-offs?

<div style="page-break-after: always;"></div>

### C. Contributions

This study makes the following contributions:

1. A systematic characterization of the IP-Adapter uniform scale parameter across the full $[0.0, 1.0]$ range with three seeds per condition, generating 51 images that map the nonlinear fidelity-variation trade-off.
2. An exploration of 46 block-level IP-Adapter configurations spanning layer selection, block selection, scale intensity, alternative prompts, and alternative reference images for content-style disentanglement.
3. A 34-image investigation of text-image conditioning interaction across five sub-experiments examining prompt specificity, semantic compatibility, and uniform versus block-level comparison.
4. A quantitative ASR evaluation comparing four models (wav2vec 2.0, Whisper-tiny, Whisper-small, Whisper-medium) on 20 MINDS-14 samples with WER/CER metrics, plus a 40-sample sampling-rate mismatch analysis.
5. A 20-file Bark TTS analysis across repeated generations, text length variation, domain complexity, and round-trip TTS-to-ASR evaluation.
6. A seed sensitivity analysis across 8 seeds for three generative paradigms (24 total outputs).
7. A cross-modal comparison framework relating vision and audio generative paradigms on controllability, stochasticity, computational cost, and evaluation methodology.

### D. Report Organization

Section II reviews the relevant literature. Section III describes the experimental methodology. Sections IV through X present results for each experiment. Section XI provides a cross-experiment discussion addressing all seven research questions. Section XII concludes the study, and Section XIII identifies future work directions.

---

<div style="page-break-after: always;"></div>

## II. Literature Review

### A. Latent Diffusion Models and SDXL Foundations

Denoising Diffusion Probabilistic Models (DDPMs), formalized by Ho et al. [6], learn to reverse a gradual noising process to generate data. The forward process adds Gaussian noise over $T$ timesteps according to a variance schedule $\beta_t$:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{1 - \beta_t}\, \mathbf{x}_{t-1},\, \beta_t \mathbf{I})$$

The reverse process learns a noise-prediction network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ trained with the simplified objective $\mathcal{L} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2]$ [6]. Latent Diffusion Models (LDMs) [7] improve computational efficiency by operating in a compressed latent space $\mathbf{z} = \mathcal{E}(\mathbf{x})$ provided by a pretrained VAE, reducing spatial dimensions by a factor of 8 while preserving generation quality. Conditioning is introduced through cross-attention layers in the U-Net [8], where latent features serve as queries and conditioning embeddings (e.g., CLIP text encodings [9]) provide keys and values.

Stable Diffusion XL (SDXL) [3] advances this framework with a dual text encoder architecture (OpenAI CLIP ViT-L/14 and OpenCLIP ViT-bigG/14) producing a 2048-dimensional conditioning signal, a 2.6-billion-parameter U-Net, and micro-conditioning on image resolution metadata. Classifier-Free Guidance (CFG) [10] amplifies prompt adherence at inference time via the modified noise prediction:

$$\hat{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \varnothing) + w \left[\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \varnothing)\right]$$

where $w$ is the guidance scale and $\varnothing$ denotes the null conditioning embedding.

### B. IP-Adapter Architecture and Decoupled Cross-Attention

IP-Adapter [2] enables image-prompted generation by introducing a parallel cross-attention pathway for CLIP image embeddings alongside the existing text cross-attention. The combined output at each cross-attention layer is:

$$\mathbf{Z} = \text{Attention}(\mathbf{Q}, \mathbf{K}_{\text{text}}, \mathbf{V}_{\text{text}}) + \lambda \cdot \text{Attention}(\mathbf{Q}, \mathbf{K}_{\text{image}}, \mathbf{V}_{\text{image}})$$

where $\lambda$ is the IP-Adapter scale controlling image conditioning strength. The image cross-attention layers have their own learned key/value projection matrices ($\mathbf{W}_K^{\text{image}}, \mathbf{W}_V^{\text{image}}$), trained while the base U-Net parameters remain frozen. A small image projection network maps the CLIP image embedding to the dimensionality expected by these projections: $\mathbf{c}_{\text{image}} = \text{Proj}(\text{CLIP}_{\text{image}}(\mathbf{x}_{\text{ref}}))$. This decoupled design preserves the base model's text-conditioning capability while adding image conditioning as a parallel, independently scalable channel.

### C. Hierarchical U-Net Features and Content-Style Disentanglement

The U-Net architecture processes features at multiple spatial scales, with a well-established property that different layers encode different levels of abstraction: deep layers (low spatial resolution, near the bottleneck) encode global structure and composition, while shallow upsampling layers encode fine-grained stylistic details such as color palettes, textures, and tonal qualities [8]. GA18C exploits this property by replacing the uniform IP-Adapter scale with a per-block scale dictionary that selectively applies image conditioning to specific U-Net upsampling blocks. By restricting IP-Adapter influence to deep decoder blocks, the reference image's style is transferred without overriding the text prompt's content control — achieving content-style disentanglement.

### D. Automatic Speech Recognition: wav2vec 2.0, Whisper, and Audio Preprocessing

wav2vec 2.0 [11] is a self-supervised speech representation framework that pre-trains on unlabeled audio via contrastive learning over quantized latent representations, then fine-tunes with a Connectionist Temporal Classification (CTC) [12] head for ASR. The CTC objective marginalizes over all valid alignments:

$$P(\mathbf{y} | \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} P(\pi_t | \mathbf{x})$$

Whisper [13] adopts an alternative encoder-decoder architecture trained on 680,000 hours of weakly supervised multilingual data, using log-mel spectrogram features and an autoregressive decoder. Whisper's scale and diversity yield strong zero-shot generalization without domain-specific fine-tuning. Both architectures require audio at 16 kHz; providing audio at an incorrect sampling rate causes temporal structure misinterpretation that severely degrades transcription quality [1].

The standard ASR evaluation metrics are Word Error Rate (WER) and Character Error Rate (CER):

$$\text{WER} = \frac{S + D + I}{N}$$

where $S$, $D$, $I$ are substitutions, deletions, and insertions, and $N$ is the reference word count [12].

### E. Text-to-Speech Synthesis with Bark and Neural Audio Codecs

Bark [5] is a multi-stage autoregressive TTS model operating in the discrete token space of a neural audio codec. Its architecture cascades three transformer stages:

$$\text{Text} \xrightarrow{\text{Stage 1}} \text{Semantic tokens} \xrightarrow{\text{Stage 2}} \text{Coarse acoustic tokens} \xrightarrow{\text{Stage 3}} \text{Fine acoustic tokens} \xrightarrow{\text{Codec}} \text{Waveform}$$

EnCodec [14] provides the underlying audio tokenization via Residual Vector Quantization (RVQ), which iteratively quantizes residual errors across multiple codebook levels. Coarse levels capture dominant spectral structure while fine levels encode residual acoustic detail. The `bark-small` variant uses smaller transformers at each stage, trading audio quality for reduced memory and faster inference. The model outputs audio at 24,000 Hz.

### F. Cross-Modal Framing for Multi-Modal Generative AI

The vision and audio paradigms studied here share several unifying themes: hierarchical discrete representations (VAE latent space for images, RVQ token hierarchy for audio), cross-attention as the universal conditioning mechanism, and self-supervised pre-training as the foundation for downstream capability (CLIP for vision-language alignment, wav2vec 2.0 for speech representations). However, they differ fundamentally in control mechanism (continuous scale parameter vs. discrete text input), stochasticity profile (seed-dependent noise initialization vs. multi-stage autoregressive sampling), and evaluation methodology (qualitative visual assessment vs. quantitative WER/CER metrics). These parallels and contrasts motivate the cross-modal analysis in this study.

---

## III. Methodology

### A. Experimental Infrastructure and Runtime Environment

All experiments were executed on Google Colab with NVIDIA A100 GPU acceleration. Vision experiments (GA18B, GA18C) used the `StableDiffusionXLPipeline` from the Hugging Face `diffusers` library with the `stabilityai/stable-diffusion-xl-base-1.0` model in FP16 precision, paired with IP-Adapter weights from `h94/IP-Adapter` (`ip-adapter_sdxl.bin`). ASR experiments (GA19B) used the Hugging Face `transformers` pipeline API with four models: `facebook/wav2vec2-base-960h`, `openai/whisper-tiny`, `openai/whisper-small`, and `openai/whisper-medium`. TTS experiments (GA19C) used the `suno/bark-small` model via the TTS pipeline. Experiments were organized into three execution groups (A: vision, B: ASR, C: TTS) to manage GPU memory, with full console logs captured for each group.

### B. Input Sources, Datasets, and Reference Assets

Image generation experiments used two reference images from the `genaibook` utility library: `SampleURL.ItemsVariation` (a photograph of household items, used as the default for GA18B) and `SampleURL.Mamoeiro` (a photograph of a papaya tree, used as the default for GA18C). All reference images were resized to 1024×1024 pixels matching SDXL's native resolution. ASR experiments used the MINDS-14 dataset [4] (`PolyAI/minds14`, `en-AU` subset), a collection of banking-intent spoken utterances in Australian English. A fixed subset of 20 samples (selected with seed 42) was used for multi-model comparison, and 10 samples for sampling-rate mismatch analysis. TTS experiments used five text inputs spanning different lengths and linguistic domains: the default GA19C text about ladybugs, a short greeting, a pangram sentence, a conversational request, a technical description, and a proper-noun sentence.

### C. Experiment Map and Parameter Sweep Design

Seven experiments were conducted, each targeting a specific research question:

| Experiment | Script(s) | Primary RQ | Key Variables | Outputs |
|-----------|-----------|-----------|---------------|---------|
| 1. Baseline Reproduction | GA18B, GA18C, GA19B, GA19C | — | Default parameters | 4 outputs |
| 2. Uniform Scale Sweep | GA18B | RQ1 | $\lambda \in [0.0, 1.0]$, reference images, inference steps | 51 images |
| 3. Block-Level Configs | GA18C | RQ2 | Layer selection, block selection, scale intensity, prompts | 46 images |
| 4. Text-Image Interaction | GA18B + GA18C | RQ3 | Scale, prompt specificity, semantic compatibility | 34 images |
| 5. ASR Model Comparison | GA19B | RQ4 | ASR model, sampling rate | 160 transcriptions |
| 6. TTS Consistency | GA19C | RQ5 | Text input, repetitions, round-trip evaluation | 20 audio files |
| 7. Seed Sensitivity | GA18B, GA18C, GA19C | RQ6 | Random seeds (8 per paradigm) | 24 outputs |

All stochastic generation calls used explicit random seeds via `torch.Generator` for reproducibility. Every parameter for every generation or inference call was logged to CSV files with timestamps.

### D. Evaluation Strategy and Recorded Outputs

Vision experiments were evaluated qualitatively through comparative visual grids and narrative description of observed patterns. For each image, generation time was recorded. ASR experiments were evaluated quantitatively using WER and CER computed by the `jiwer` library, comparing predicted transcriptions against ground-truth annotations from the MINDS-14 dataset. TTS experiments were evaluated through audio duration, waveform RMS amplitude, generation time, and round-trip WER (generating speech with Bark, then transcribing with the best-performing ASR model). All outputs were saved to structured directories under `hw13_experiments/` with standardized naming conventions.

### E. Reproducibility, Logging, and Artifact Management

Every experiment logged full parameters, seeds, output paths, generation times, and timestamps to per-experiment CSV files. A consolidated `summary_statistics.json` records aggregate metrics: 149 total images generated, 29 audio files, 164 transcriptions, and 828.2 seconds of total GPU time. Console output from all three execution groups was captured to log files in `hw13_printouts/`. The orchestrator script (`kamp_hw13.py`) managed experiment sequencing, parameter injection, and artifact collection, while individual experiment logic was encapsulated in helper modules (`hw13_experiment_runner.py`, `hw13_data_utils.py`).

---

<div style="page-break-after: always;"></div>

## IV. Results: Experiment 1 - Baseline Reproduction

### A. Objective and Baseline Conditions

Experiment 1 reproduced the default behavior of all five scripts with seed 42 to establish reference outputs for subsequent parameter variation experiments. All controlled variables used the default values documented in the original scripts.

### B. Baseline Outputs by Script

#### 1. GA18B - Uniform-Scale Image Variation Baseline

The GA18B baseline generated an image variation of the `ItemsVariation` reference image using IP-Adapter with uniform scale $\lambda = 0.8$, an empty text prompt, and 25 inference steps. Generation time was 3.428 seconds (elevated due to initial pipeline warmup; subsequent generations averaged 1.58 seconds). The output demonstrated IP-Adapter's ability to produce a semantically coherent variation of the reference image — preserving high-level content and composition while introducing natural variation in fine detail.

#### 2. GA18C - Block-Level Style-Conditioned Generation Baseline

The GA18C baseline generated a style-conditioned image using the block-level scale configuration `{"up": {"block_0": [0.0, 1.0, 0.0]}}` with the text prompt "a cat inside of a box" and the `Mamoeiro` (papaya tree) reference image as the style source. Generation time was 1.420 seconds. The output demonstrated content-style disentanglement: the semantic content matched the text prompt while the visual style reflected the warm, tropical character of the reference image.

#### 3. GA19A - Audio Dataset Inspection Baseline

GA19A performed dataset exploration of the MINDS-14 `en-AU` subset, confirming the dataset structure including audio waveforms, sampling rates, intent classes, and transcription annotations. This script does not produce generated outputs but validated the data pipeline for subsequent ASR experiments.

#### 4. GA19B - Baseline Automatic Speech Recognition Output

The GA19B baseline transcribed a MINDS-14 utterance using the default `facebook/wav2vec2-base-960h` model at 16 kHz sampling rate. The ground-truth transcription described an issue with an online banking app. The predicted transcription exhibited substantial word-level errors (WER = 0.459), particularly with the Australian English accent and conversational speech patterns. Inference time was 1.153 seconds for a 22.3-second audio clip.

#### 5. GA19C - Baseline Bark Text-to-Speech Output

The GA19C baseline synthesized speech from the default text ("Ladybugs have had important roles in culture and religion...") using `suno/bark-small` with seed 42. The generated audio was 8.587 seconds at 24,000 Hz with an RMS amplitude of 0.023. Generation time was 21.643 seconds — approximately 2.5 times the audio duration — reflecting the computational cost of Bark's three-stage autoregressive pipeline.

### C. Baseline Result Synthesis

The baselines confirmed that all five pipelines produce valid outputs with their default configurations. Key observations: (1) IP-Adapter image generation with SDXL is computationally moderate (~1.5 seconds per image after warmup), (2) Bark TTS is computationally expensive relative to the output duration, and (3) the default wav2vec 2.0 ASR model produces noticeable errors on Australian-accented conversational speech. These baselines serve as the controlled reference for all subsequent experiments.

![Figure 1: Baseline outputs from all five scripts (GA18B, GA18C, GA19A, GA19B, GA19C) generated with default parameters and seed 42.](figures/exp1_baseline_gallery.png)

*Figure 1.* Baseline output gallery showing the default-parameter results for each of the five scripts. This establishes the reference conditions against which all subsequent parameter-variation experiments are compared.

---

## V. Results: Experiment 2 - IP-Adapter Uniform Scale Sensitivity (GA18B)

### A. RQ1 Alignment: Reference Fidelity vs. Output Variation

This experiment directly addresses RQ1 by sweeping the uniform IP-Adapter scale $\lambda$ across its full $[0.0, 1.0]$ range to characterize the relationship between $\lambda$ and the fidelity of the generated output to the reference image.

### B. Experimental Conditions and Scale Sweep Structure

The experiment comprised three sub-experiments totaling 51 images:

- **Sub-experiment 2A (Primary sweep):** 11 scale values $\lambda \in \{0.0, 0.1, 0.2, \ldots, 1.0\}$ with three seeds (42, 123, 456) per value and an empty text prompt, using the `ItemsVariation` reference image. Total: 33 images.
- **Sub-experiment 2B (Alternative reference):** 6 scale values $\lambda \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ with two seeds (42, 123) using the `Mamoeiro` reference image. Total: 12 images.
- **Sub-experiment 2C (Inference steps variation):** Three step counts $\{10, 25, 50\}$ at fixed $\lambda = 0.8$ with two seeds. Total: 6 images.

### C. Quantitative Trends and Qualitative Comparisons

**Generation time:** Mean generation time across the primary sweep was approximately 1.55 seconds per image, with minimal variation across scale values (1.416 seconds at $\lambda = 0.0$ to 1.597 seconds at $\lambda = 0.5$). The IP-Adapter scale does not appreciably affect computational cost, confirming that the adapter's inference overhead is constant regardless of scale setting. Sub-experiment 2C showed that inference steps directly impact generation time: 10 steps averaged faster execution while 50 steps averaged 1.72 seconds, consistent with the linear scaling of denoising iterations.

**Scale-fidelity relationship:** Visual inspection of the 33-image primary sweep grid reveals a nonlinear transition. At $\lambda = 0.0$, outputs are unconditioned generations bearing no resemblance to the reference image — effectively random SDXL outputs guided only by the null text embedding. From $\lambda = 0.1$ to $\lambda = 0.3$, reference image characteristics begin to emerge subtly, with outputs showing loose thematic similarity but substantial creative divergence. The range $\lambda \in [0.4, 0.6]$ represents the most perceptually dynamic regime, where outputs maintain clear semantic connection to the reference while exhibiting meaningful variation in composition and detail. At $\lambda \geq 0.7$, outputs converge increasingly toward the reference image, with $\lambda = 1.0$ producing outputs that closely resemble the reference in content, composition, and color palette while remaining distinct generated images rather than pixel-level copies.

**Reference image generalization (Sub-experiment 2B):** The scale-fidelity relationship generalizes across reference images. The `Mamoeiro` reference produced analogous transitions, with the tropical scene's characteristic warm greens and organic forms emerging at moderate scales and dominating at high scales. This confirms that the observed nonlinear transition is a property of the IP-Adapter mechanism rather than an artifact of a specific reference image.

**Inference steps effect (Sub-experiment 2C):** At the fixed scale of $\lambda = 0.8$, all three step counts produced outputs with strong reference fidelity. The 10-step condition exhibited slightly reduced fine-detail quality, while the 25-step and 50-step conditions produced comparable results, suggesting that 25 steps provides a sufficient denoising trajectory for IP-Adapter-conditioned generation.

![Figure 2: Grid of GA18B outputs across IP-Adapter uniform scale values from 0.0 to 1.0 with multiple seeds.](figures/ga18b_scale_sweep_grid.png)

**Figure 2.** An 11×3 grid of GA18B outputs spanning IP-Adapter scales $\lambda = 0.0$ to $1.0$ (rows) across three seeds (42, 123, 456). At $\lambda = 0.0$, outputs are diverse and unrelated to the reference; as $\lambda$ increases, images progressively converge toward the coffee-cup flat-lay reference, with strong compositional and color fidelity emerging by $\lambda \geq 0.5.$

### D. Interpretation Relative to RQ1

The results confirm the hypothesized nonlinear relationship between $\lambda$ and reference fidelity. The IP-Adapter scale provides a continuous, monotonic control from unconditioned generation ($\lambda = 0$) to reference-guided synthesis ($\lambda = 1$), with the most compositionally interesting variation concentrated in the $[0.4, 0.7]$ range. The transition does not exhibit abrupt thresholds but rather a smooth gradient with the steepest perceptual change occurring between $\lambda = 0.2$ and $\lambda = 0.5$. The relationship generalizes across reference images and is robust to inference step variation.

---

<div style="page-break-after: always;"></div>

## VI. Results: Experiment 3 - Block-Level Scale Configurations for Content-Style Disentanglement (GA18C)

### A. RQ2 Alignment: Block Selection and Style Transfer Behavior

This experiment addresses RQ2 by systematically exploring alternative block-level IP-Adapter configurations beyond the default, characterizing how different block and layer selections affect the type and degree of style transfer while preserving text-driven content control.

### B. Experimental Conditions and Block-Configuration Variants

The experiment comprised five sub-experiments totaling 46 images, all using the text prompt "a cat inside of a box" and the `Mamoeiro` reference image unless otherwise noted:

- **Sub-experiment 3A (Layer variation within block 0):** Six layer configurations within `block_0` — `[1,0,0]`, `[0,1,0]` (default), `[0,0,1]`, `[1,1,0]`, `[0,1,1]`, `[1,1,1]` — with three seeds each. Total: 18 images.
- **Sub-experiment 3B (Block selection comparison):** The middle-layer configuration `[0,1,0]` applied to `block_0`, `block_1`, and `block_2` with two seeds each. Total: 6 images.
- **Sub-experiment 3C (Scale intensity variation):** Five scale values $s \in \{0.25, 0.5, 0.75, 1.0, 1.5\}$ for the active layer within `block_0` with two seeds each. Total: 10 images.
- **Sub-experiment 3D (Alternative text prompts):** Four prompts ("a cat inside of a box", "a mountain landscape at sunset", "a portrait of an old man", "a futuristic cityscape") with the default block configuration and two seeds each. Total: 8 images.
- **Sub-experiment 3E (Alternative reference image):** Two reference images (`Mamoeiro`, `ItemsVariation`) with the default block configuration and two seeds each. Total: 4 images.

Generation time was consistently 1.44 seconds across all configurations, confirming that block-level scale selection does not affect computational cost.

### C. Configuration-by-Configuration Findings

**Layer variation (3A):** Different cross-attention layers within `block_0` produced measurably different style transfer characteristics. Single-layer activations (`[1,0,0]`, `[0,1,0]`, `[0,0,1]`) each transferred recognizable stylistic elements from the reference image — warm tropical tones, organic textures — while preserving the text prompt's "cat in a box" content. Multi-layer activations (`[1,1,0]`, `[0,1,1]`, `[1,1,1]`) produced progressively stronger style transfer, with the all-layers-active configuration `[1,1,1]` exhibiting the most pronounced reference image influence. This confirms that individual cross-attention layers within a block make distinct contributions to style representation.

**Block selection (3B):** Applying the same `[0,1,0]` configuration to different upsampling blocks revealed clear differences in style transfer character. `block_0` (deepest, lowest resolution) transferred the most global stylistic attributes — overall color temperature and tonal character. `block_1` (intermediate resolution) modulated style at a somewhat more localized level. `block_2` (shallowest, highest resolution) had the least visible style transfer effect, suggesting that high-resolution upsampling layers encode content-specific spatial detail rather than transferable stylistic attributes. This gradient confirms the hierarchical feature encoding hypothesis.

**Scale intensity (3C):** The block-level scale parameter acts as a continuous dial for style transfer intensity. At $s = 0.25$, style transfer was subtle — the reference image's warm color palette was faintly present. At $s = 0.75$, the tropical character was clearly evident. At $s = 1.5$ (supermaximal), the reference image's influence was strong enough to partially compete with the text prompt's content specification, producing outputs where stylistic elements began to modify semantic content. The $s \in [0.5, 1.0]$ range represented the optimal balance for disentangled style transfer.

**Prompt robustness (3D):** The content-style disentanglement was robust across semantically diverse text prompts. All four prompts produced outputs with appropriate semantic content (landscapes, portraits, cityscapes) rendered in the warm tropical style of the reference image. This confirms that the block-level configuration successfully decouples style from content across varied semantic domains.

**Reference image generalization (3E):** Replacing the `Mamoeiro` reference with `ItemsVariation` changed the transferred stylistic attributes accordingly — outputs exhibited the photographic quality and color characteristics of the household items image rather than the tropical warmth. The content-style disentanglement mechanism generalizes across different style source images.

![Figure 3: Comparison of GA18C outputs across different block-level IP-Adapter configurations.](figures/ga18c_block_comparison.png)

**Figure 3.** A 3×2 grid comparing block-level IP-Adapter conditioning across three upsampling blocks (block_0, block_1, block_2) and two seeds, demonstrating that earlier blocks transfer stronger, more global stylistic attributes from the reference image, while later blocks yield progressively weaker or more localized conditioning.

### D. Interpretation Relative to RQ2

Different block-level configurations produce meaningfully different style transfer outcomes. The key findings are: (1) individual cross-attention layers within a block make distinct stylistic contributions, with multi-layer activation producing additive effects; (2) deeper upsampling blocks transfer more global stylistic attributes while shallower blocks have diminishing style transfer impact; (3) the block-level scale provides continuous intensity control; and (4) the content-style disentanglement is robust across diverse text prompts and reference images. The default configuration `{"up": {"block_0": [0.0, 1.0, 0.0]}}` represents one well-balanced point in a rich design space.

---

## VII. Results: Experiment 4 - Text-Image Conditioning Interaction (GA18B + GA18C)

### A. RQ3 Alignment: Joint Effects of Text and Image Guidance

This experiment addresses RQ3 by examining what happens when uniform IP-Adapter image conditioning (as in GA18B) is combined with an active text prompt, and by directly comparing uniform versus block-level conditioning under matched conditions.

### B. Prompt and Image-Conditioning Comparison Structure

Five sub-experiments totaling 34 images:

- **Sub-experiment 4A (Scale sweep with text prompt):** $\lambda \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ with the prompt "a beautiful sunset over the ocean" and `ItemsVariation` reference, two seeds per condition. Total: 12 images.
- **Sub-experiment 4B (Prompt specificity comparison):** Three prompts of varying specificity — minimal ("scene"), moderate ("a beautiful sunset over the ocean"), and detailed (a long photorealistic description) — at fixed $\lambda = 0.5$, two seeds each. Total: 6 images.
- **Sub-experiment 4C (Semantic compatibility):** Three prompts varying in compatibility with the `ItemsVariation` reference — compatible ("household items on a table"), incompatible ("a mountain landscape at sunset"), and neutral ("an abstract painting") — at $\lambda = 0.5$, two seeds each. Total: 6 images.
- **Sub-experiment 4D (Block-level with varied prompts):** Two prompts under the default block-level configuration with the `Mamoeiro` reference, two seeds each. Total: 4 images (excluded from final count due to overlap with 4E).
- **Sub-experiment 4E (Uniform vs. block-level direct comparison):** Uniform ($\lambda = 0.5$) versus block-level (default) conditioning with the prompt "a cat inside of a box" and `Mamoeiro` reference, three seeds each. Total: 6 images.

<div style="page-break-after: always;"></div>

### C. Interaction Findings Across Conditions

**Scale-prompt balance (4A):** The gradual transition from text-dominated to image-dominated output was clearly visible. At $\lambda = 0.0$, outputs depicted sunset ocean scenes consistent with the text prompt, with no reference image influence. At $\lambda = 0.2$, subtle reference image characteristics appeared alongside the text-driven content. At $\lambda = 0.4$–$0.6$, both modalities contributed visibly — outputs contained elements of both the sunset scene and the household items reference. At $\lambda = 0.8$–$1.0$, the reference image dominated and the sunset prompt had minimal visible effect. Generation times increased slightly with scale (1.428 seconds at $\lambda = 0.0$ to 1.618 seconds at $\lambda = 1.0$), reflecting the marginal computational overhead of active IP-Adapter cross-attention.

**Prompt specificity (4B):** At the moderate scale of $\lambda = 0.5$, more specific text prompts maintained greater influence over the output. The minimal prompt ("scene") produced outputs heavily influenced by the reference image, while the detailed photorealistic prompt competed more effectively against the image conditioning, producing outputs that blended specific textual elements (vivid sky colors, ocean detail) with the reference image's characteristics. This suggests that prompt specificity quantitatively affects the text-image balance at any given scale.

**Semantic compatibility (4C):** Semantically compatible prompts produced more coherent outputs than incompatible ones. The compatible prompt ("household items on a table") created a harmonious blend with the `ItemsVariation` reference, while the incompatible prompt ("a mountain landscape at sunset") produced outputs with visible tension between the two conditioning signals. The neutral prompt ("an abstract painting") allowed the IP-Adapter to exert moderate influence without semantic conflict.

**Uniform vs. block-level comparison (4E):** Under matched conditions (same prompt, reference image, and seeds), uniform scaling at $\lambda = 0.5$ applied the reference image's influence holistically — affecting both content and style. Block-level scaling applied the reference image's influence selectively to style while preserving the text prompt's content specification. This is the fundamental distinction: uniform scaling blends content and style from the reference, while block-level scaling separates them.

![Figure 4: Direct comparison of uniform-scale versus block-level IP-Adapter conditioning under matched conditions.](figures/uniform_vs_block_comparison.png)

**Figure 4.** A 2×3 grid comparing uniform (GA18B) and block-level (GA18C) conditioning across three seeds for a cat-in-box prompt, showing that uniform conditioning applies the reference style broadly with flat, illustrative aesthetics, while block-level conditioning achieves better content–style disentanglement with more realistic subject rendering and localized style transfer.

### D. Interpretation Relative to RQ3

The text and image conditioning signals interact additively as predicted by the decoupled cross-attention architecture: $\mathbf{Z} = \mathbf{Z}_{\text{text}} + \lambda \cdot \mathbf{Z}_{\text{image}}$. At low $\lambda$, the text prompt dominates; at high $\lambda$, the image conditioning dominates; in between, both contribute. The transition is gradual rather than abrupt. Prompt specificity modulates the effective text contribution — more detailed prompts maintain influence at higher scales. Semantic compatibility between text and image produces more coherent outputs than conflicting pairings. The uniform vs. block-level distinction is the most important structural finding: block-level scaling provides a principled mechanism for style-only transfer that uniform scaling cannot achieve.

---

## VIII. Results: Experiment 5 - ASR Model Comparison and Sampling-Rate Robustness (GA19B)

### A. RQ4 Alignment: Model Choice and Preprocessing Robustness

This experiment addresses RQ4 through a quantitative comparison of four ASR models on correctly resampled audio and a robustness analysis under sampling rate mismatch.

### B. ASR Models, Audio Conditions, and Evaluation Metrics

- **Sub-experiment 5A (Multi-model comparison):** Four models — `facebook/wav2vec2-base-960h`, `openai/whisper-tiny`, `openai/whisper-small`, `openai/whisper-medium` — evaluated on 20 MINDS-14 samples at the correct 16 kHz sampling rate. Total: 80 transcriptions.
- **Sub-experiment 5B (Sampling rate mismatch):** Two models (wav2vec 2.0 and Whisper-small) evaluated on 40 samples at 8 kHz (native, no resampling). Total: 80 transcriptions.

All transcriptions were evaluated with WER and CER computed by the `jiwer` library against ground-truth annotations.

### C. Accuracy, Error, and Runtime Findings

**Multi-model comparison at 16 kHz (5A):**

| Model | Mean WER | WER Std | Mean CER | Mean Inference Time (s) |
|-------|----------|---------|----------|------------------------|
| wav2vec2-base-960h | 0.4074 | 0.3384 | 0.2110 | 0.0299 |
| whisper-tiny | 0.2586 | 0.2670 | 0.1448 | 0.1787 |
| whisper-small | 0.1797 | 0.2331 | 0.1026 | 0.4166 |
| whisper-medium | 0.1715 | 0.2246 | 0.1000 | 0.7407 |

Whisper-medium achieved the lowest mean WER (0.172), followed closely by Whisper-small (0.180). Whisper-tiny performed notably better than wav2vec 2.0 despite being substantially smaller than the small and medium Whisper variants. wav2vec 2.0 had the highest error rate (0.407 WER) but was dramatically faster at 0.030 seconds mean inference time — approximately 25 times faster than Whisper-medium. The WER standard deviations were high for all models (0.22–0.34), reflecting the heterogeneity of the MINDS-14 samples, which include both short, clear utterances (some achieving 0.0 WER) and longer, accented conversational speech (WER exceeding 0.6).

![Figure 5: Bar chart comparing mean WER across four ASR models on MINDS-14 at 16 kHz.](figures/asr_model_comparison_bar.png)

**Figure 5.** Mean WER bar chart ranked by performance: Whisper-medium achieves the lowest mean WER (0.172), followed by Whisper-small (0.180), Whisper-tiny (0.259), and wav2vec2-base-960h (0.407), demonstrating that larger Whisper variants substantially outperform wav2vec 2.0 on this conversational banking-domain dataset.

**Sampling rate mismatch at 8 kHz (5B):**

| Model | Correct SR (16 kHz) WER | Mismatched SR (8 kHz) WER | WER Increase |
|-------|-------------------------|---------------------------|-------------|
| wav2vec2-base-960h | 0.4074 | 0.7470 | +0.340 (83% relative) |
| whisper-small | 0.1797 | 0.4119 | +0.232 (129% relative) |

Providing audio at 8 kHz without resampling (i.e., processed as if it were 16 kHz) produced severe degradation for both architectures. wav2vec 2.0's WER increased from 0.407 to 0.747, and Whisper-small's WER increased from 0.180 to 0.412. The CER degradation was even more dramatic (wav2vec 2.0: 0.211 → 0.533; Whisper-small: 0.103 → 0.344). Both models interpreted the 8 kHz audio as if it were playing at half speed with halved frequency content, causing systematic temporal and spectral misinterpretation.

![Figure 6: Effect of sampling rate mismatch on ASR WER for wav2vec 2.0 and Whisper-small.](figures/asr_sampling_rate_effect.png)

**Figure 6.** WER as a function of input sampling rate for wav2vec2-base-960h and Whisper-small across 8, 16, 22, and 48 kHz. Both models achieve optimal performance at 16 kHz (their expected native rate), with severe degradation at 8 kHz (wav2vec2 WER rises to ~0.98) and 48 kHz (both models approach WER ~1.0); Whisper-small demonstrates greater robustness to sampling rate mismatch overall.

### D. Interpretation Relative to RQ4

Whisper models substantially outperform wav2vec 2.0 on the MINDS-14 banking-domain speech, confirming the hypothesis that the encoder-decoder architecture with attention-based decoding and massive weakly supervised training data provides superior accuracy on conversational, accented speech. Performance improves monotonically with model size across Whisper variants (tiny < small < medium), though the marginal gain from small to medium is modest (0.180 → 0.172) while the computational cost more than doubles. Sampling rate mismatch produces severe degradation for both architectures — this is not a graceful degradation but a near-catastrophic failure, confirming that correct audio preprocessing is a critical prerequisite for ASR deployment. Interestingly, the relative WER increase was larger for Whisper-small (129%) than for wav2vec 2.0 (83%), though Whisper-small's absolute error rate under mismatch (0.412) was still substantially better than wav2vec 2.0's performance even at the correct sampling rate (0.407).

---

## IX. Results: Experiment 6 - Bark TTS Consistency and Variation (GA19C)

### A. RQ5 Alignment: Repeated Generation Consistency and Text Effects

This experiment addresses RQ5 by examining Bark's output variation across repeated generations with different seeds and across text inputs of varying length, complexity, and domain.

### B. Text Categories, Repetition Structure, and Output Capture

Four sub-experiments totaling 20 audio files:

- **Sub-experiment 6A (Repeated generation):** The default ladybug text generated with 5 seeds (42, 123, 456, 789, 1024).
- **Sub-experiment 6B (Text length variation):** Three texts — short ("Hello, how are you?", ~5 words), medium (ladybug text, ~17 words), and long (pangram text, ~30 words) — with 2 seeds each. Total: 6.
- **Sub-experiment 6C (Text domain/complexity):** Conversational, technical, and proper-noun texts with 2 seeds each. Total: 6.
- **Sub-experiment 6D (Round-trip TTS→ASR):** Three texts evaluated via round-trip WER. Total: 3.

### C. Variation, Audio Quality, and Round-Trip Findings

**Repeated generation consistency (6A):** Five generations of the same 17-word text with different seeds produced audio durations ranging from 7.43 to 12.93 seconds (mean 9.85, std 2.09), representing a 74% range relative to the shortest output. RMS amplitude varied from 0.023 to 0.143, a six-fold range. Generation times ranged from 17.48 to 29.91 seconds (mean 22.95). This substantial variation across seeds confirms that Bark's three-stage autoregressive sampling introduces appreciable stochastic variability in both timing and amplitude characteristics. The semantic content (words spoken) remained consistent, but prosodic realization — speaking rate, intonation contours, emphasis patterns — varied meaningfully.

**Text length variation (6B):** There is a clear positive relationship between text length and both audio duration and generation time. Short texts (5 words) produced audio of 3.17 seconds mean duration with 7.45 seconds mean generation time. Medium texts (17 words) produced 8.01 seconds audio with 18.58 seconds generation time. Long texts (30 words) produced 13.79 seconds audio with 31.89 seconds generation time. The generation-time-to-audio-duration ratio was approximately 2.3–2.4× across all lengths, indicating roughly linear computational scaling with text length.

**Text domain and complexity (6C):** Conversational text produced audio of 5.73 seconds mean duration with moderate variation (std 0.65). Technical text produced 7.33-second audio. Proper-noun text ("Professor Schwarzenegger from the University of Cincinnati...") produced 6.29-second audio with notably lower RMS amplitude (0.026 mean), suggesting that the model's uncertainty with unusual proper nouns manifested as reduced prosodic confidence. Technical vocabulary was handled adequately, though the narrow RMS range (0.032 mean) suggests more monotone delivery compared to the conversational text.

<div style="page-break-after: always;"></div>

**Round-trip evaluation (6D):** Three texts were processed through the TTS→ASR pipeline (Bark generation followed by ASR transcription):

| Text Category | Round-Trip WER |
|---------------|---------------|
| Short ("Hello, how are you?") | 0.250 |
| Technical (CNN description) | 0.231 |
| Proper nouns (Professor Schwarzenegger...) | 0.000 |

The proper-noun text achieved perfect round-trip WER (0.0), indicating that the generated speech was fully intelligible and phonetically accurate for this input. The short and technical texts showed moderate round-trip WER (0.23–0.25), which may reflect either Bark's output quality variations or the ASR model's limitations on synthesized speech.

![Figure 7: Bark TTS generation time as a function of text input length and category.](figures/tts_generation_time_scaling.png)

**Figure 7.** Scatter plot of Bark generation time versus input word count across six text categories (short, medium, long, conversational, proper nouns, and technical), showing a clear positive correlation. Short inputs (~4 words) generate in approximately 5–10 seconds, while long inputs (~30 words) require 29–35 seconds, with intermediate categories clustering in the 12–20 second range.

![Figure 8: Round-trip TTS-to-ASR WER for three text categories processed through the Bark-to-Whisper pipeline.](figures/tts_round_trip_wer.png)

**Figure 8.** Bar chart of round-trip TTS→ASR word error rate (WER) by text category, revealing that short texts exhibit the highest WER (0.250), followed closely by technical texts (0.231), while proper nouns achieve perfect round-trip fidelity (WER = 0.000). This suggests that Bark produces clearer, more recognizable speech for proper noun inputs than for short or technical phrases.

### D. Interpretation Relative to RQ5

Bark produces appreciable variation across repeated generations — consistent semantic content with varying prosodic realization — confirming the inherently stochastic nature of the multi-stage autoregressive architecture. Audio duration variability of $\pm$21% (std/mean) indicates that Bark's speaking rate is not tightly controlled. Generation time scales approximately linearly with text length, with a consistent ratio of ~2.3× real-time audio duration. Text domain affects delivery characteristics: conversational text produces more dynamic prosody, while technical and proper-noun texts produce more measured delivery. The round-trip evaluation demonstrates that Bark's output is generally intelligible, with perfect reconstruction achieved for one test case, though moderate errors persist for others.

---

<div style="page-break-after: always;"></div>

## X. Results: Experiment 7 - Seed Sensitivity and Reproducibility

### A. RQ6 Alignment: Seed Effects Across Vision and Audio Outputs

This experiment addresses RQ6 by comparing seed sensitivity across three generative paradigms under fixed default parameters, testing whether conditioning strength and modality affect the degree of seed-dependent variation.

### B. Seed Sweep Structure and Comparison Targets

Eight seeds (42, 123, 456, 789, 1024, 2048, 3000, 4096) were applied to each of three paradigms:

- **GA18B (vision, uniform scale):** $\lambda = 0.8$, empty prompt, `ItemsVariation` reference, 25 steps. 8 images.
- **GA18C (vision, block-level):** Default block config, "a cat inside of a box", `Mamoeiro` reference, 25 steps. 8 images.
- **GA19C (Bark TTS):** Default ladybug text, `bark-small`. 8 audio files.

Total: 24 outputs. GA19B (ASR) was excluded because the ASR pipeline is deterministic given the same model and audio input.

### C. Reproducibility Findings and Stability Patterns

**Vision — uniform scale (GA18B):** Mean generation time 1.550 seconds with very low variation (std 0.043). Across eight seeds at $\lambda = 0.8$, the generated images maintained strong semantic consistency with the reference image — all outputs preserved the high-level content, composition, and color palette of the `ItemsVariation` reference. Variation was limited to fine-grained details: specific object arrangements, texture patterns, and minor compositional shifts. This confirms the hypothesis that high IP-Adapter scale constrains seed-dependent variation by strongly conditioning the diffusion trajectory.

**Vision — block-level (GA18C):** Mean generation time 1.425 seconds with extremely low variation (std 0.004). The block-level configuration produced even more consistent outputs across seeds than the uniform-scale configuration. All eight outputs depicted a cat in a box with consistent tropical style transfer from the Mamoeiro reference. The combination of an active text prompt (constraining content) and block-level image conditioning (constraining style) leaves relatively little freedom for the seed to influence the output, resulting in very low seed sensitivity.

**Audio — Bark TTS (GA19C):** Mean generation time 22.465 seconds with substantial variation (std 3.889, coefficient of variation 17.3%). This is dramatically higher variation than either vision paradigm. Generation times ranged from 17.40 to 30.19 seconds — a 1.73× ratio between the fastest and slowest generation. This variation reflects the non-deterministic nature of autoregressive sampling across Bark's three transformer stages, where the seed influences the sampling trajectory at every token generation step.

![Figure 9: Seed sensitivity comparison across GA18B, GA18C, and GA19C showing generation time variability across 8 seeds per paradigm.](figures/seed_sensitivity_summary.png)

*Figure 9.* Seed sensitivity summary comparing generation time variability across eight seeds for three generative paradigms: GA18B (uniform-scale image), GA18C (block-level image), and GA19C (Bark TTS). Higher dispersion indicates greater seed sensitivity.

### D. Interpretation Relative to RQ6

Seed sensitivity varies dramatically across paradigms, confirming the hypothesis. The ranking from lowest to highest seed sensitivity is: GA18C (block-level image conditioning) < GA18B (uniform-scale image conditioning) < GA19C (Bark TTS). The vision paradigms exhibit low seed sensitivity because the conditioning signals (image embedding, text prompt) strongly constrain the diffusion process, leaving the initial noise tensor (controlled by the seed) as a secondary influence. The dual conditioning of GA18C (text + block-level image) produces the tightest constraint. Bark TTS exhibits high seed sensitivity because autoregressive sampling at each of three stages amplifies stochastic variation multiplicatively — a small difference in early token sampling cascades through subsequent stages. This has practical implications: IP-Adapter image generation is highly reproducible across seeds, making seed selection relatively unimportant for output quality, while Bark TTS outputs should be generated with multiple seeds and the best result selected.

---

<div style="page-break-after: always;"></div>

## XI. Discussion

### A. Synthesis Across Vision Experiments (RQ1-RQ3)

The three vision experiments collectively characterize the IP-Adapter conditioning design space along three axes: scale magnitude (RQ1), scale topology (RQ2), and multi-modal interaction (RQ3). A unifying theme is the additive cross-attention architecture $\mathbf{Z} = \mathbf{Z}_{\text{text}} + \lambda \cdot \mathbf{Z}_{\text{image}}$, which provides predictable, continuous control over the text-image balance. The uniform scale parameter $\lambda$ offers a single-dimensional control that simultaneously affects content and style transfer from the reference image. Block-level scaling introduces a second control dimension — spatial selectivity — that disentangles content from style by exploiting the U-Net's hierarchical feature encoding.

The practical implication is a two-tier control framework: (1) use uniform scaling when the goal is holistic image variation with controllable reference fidelity (recommended range: $\lambda \in [0.4, 0.7]$ for interesting variation), and (2) use block-level scaling when the goal is style-only transfer with independent content specification (recommended: deep blocks with $s \in [0.5, 1.0]$ for balanced transfer). The 131 images generated across Experiments 2-4 support this framework with consistent empirical evidence across multiple reference images, text prompts, and seeds.

### B. Synthesis Across Audio Experiments (RQ4-RQ5)

The audio experiments reveal a fundamental asymmetry between the recognition (ASR) and generation (TTS) sides of the speech-text loop. ASR is a deterministic discriminative task with well-established quantitative metrics (WER, CER), while TTS is a stochastic generative task where evaluation is inherently more subjective and approximate. The Whisper family's consistent superiority over wav2vec 2.0 on MINDS-14 confirms the practical advantage of large-scale weakly supervised training on diverse data for domain-general ASR. The sampling-rate mismatch analysis (WER nearly doubling for both architectures) underscores that even powerful models are fundamentally dependent on correct input preprocessing — a critical engineering lesson alongside the model-selection question.

Bark's TTS analysis reveals a system with high generative flexibility (substantial prosodic variation across seeds and diverse text handling) at the cost of considerable computational overhead (mean 19.2 seconds per generation) and limited fine-grained control. The round-trip evaluation provides a bridge between the TTS and ASR experiments, demonstrating that Bark-generated speech is sufficiently intelligible for ASR transcription in most cases, with round-trip WER ranging from 0.0 to 0.25.

### C. Reproducibility and Stability Insights (RQ6)

The seed sensitivity analysis quantifies a fundamental difference in the determinism of diffusion-based and autoregressive generative paradigms. IP-Adapter image generation with SDXL exhibits low seed sensitivity (generation time CoV: 2.8% for uniform, 0.3% for block-level), consistent with the strong conditioning constraints imposed by the CLIP image embedding and text prompt on the diffusion trajectory. Bark TTS exhibits high seed sensitivity (generation time CoV: 17.3%), reflecting the multiplicative amplification of stochastic variation across three autoregressive sampling stages.

This finding has direct implications for experimental design and practical deployment. For IP-Adapter experiments, a small number of seeds (2-3) is sufficient to capture the variability range, and results are broadly reproducible. For Bark TTS, a larger seed exploration (5+) is advisable, and users should expect meaningful output variation across generations of the same text.

### D. Cross-Modal Comparison and Trade-Off Analysis (RQ7)

The cross-modal analysis synthesizes timing, controllability, and evaluation data across all four paradigms:

| Dimension | GA18B/GA18C (Image) | GA19B (ASR) | GA19C (TTS) |
|-----------|-------------------|-------------|-------------|
| Mean time per output | 1.5 / 1.4 seconds | 0.28 seconds | 19.2 seconds |
| Controllable parameters | 5-6 | 3 | 2 |
| Stochasticity | Low-moderate | Deterministic | High |
| Evaluation metrics | Qualitative | WER/CER | Round-trip WER, subjective |
| Conditioning signal | CLIP image + text | Audio waveform | Text tokens |

The IP-Adapter image paradigm offers the richest control surface (scale, block topology, text prompt, reference image, seed) with moderate computational cost. ASR is the fastest paradigm and the only deterministic one, with the best-defined evaluation methodology. Bark TTS has the highest computational cost, the fewest controllable parameters, and the highest stochasticity — yet it achieves the remarkable feat of converting unstructured text into naturalistic continuous-domain audio, a transformation of arguably greater complexity than image-to-image variation.

A key structural insight is the relationship between control granularity and output diversity. IP-Adapter provides continuous, fine-grained control via $\lambda$ and block configurations, which constrains output diversity at high conditioning strength. Bark provides coarse control (text input and seed) with high output diversity — each generation is a distinct realization. This trade-off between controllability and diversity appears to be fundamental to the diffusion-versus-autoregressive architectural distinction.

![Figure 10: Cross-modal comparison table summarizing controllability, stochasticity, computational cost, and evaluation methodology across all paradigms.](figures/cross_modal_comparison_table_v2.png)

*Figure 10.* Cross-modal paradigm comparison summarizing mean generation time, number of controllable parameters, stochasticity level, and evaluation approach for IP-Adapter image generation (GA18B/GA18C), ASR (GA19B), and Bark TTS (GA19C).

<div style="page-break-after: always;"></div>

### E. Limitations and Validity Considerations

Several limitations should be noted. First, the vision experiments rely on qualitative evaluation — no pixel-level similarity metrics (e.g., SSIM, LPIPS) or perceptual quality scores (e.g., FID) were computed, as these would require either reference-output alignment or large sample sets not feasible within the study scope. Second, the MINDS-14 evaluation used 20 samples per model, which limits the statistical power for detecting small WER differences between Whisper-small and Whisper-medium. Third, the Bark analysis used the `bark-small` variant; the full `suno/bark` model may exhibit different quality and consistency characteristics. Fourth, all experiments were conducted on a single GPU type (A100), and timing measurements may not generalize to other hardware configurations. Fifth, the round-trip TTS→ASR evaluation provides only an indirect measure of TTS quality, as errors may originate from either the TTS or ASR stage.

---

## XII. Conclusion

### A. Summary of the HW13 Study

This study presents a systematic experimental investigation of multi-modal generative AI across vision and audio domains. Seven experiments generated 149 images, 29 audio files, and 164 ASR transcriptions, addressing seven research questions spanning IP-Adapter image conditioning, content-style disentanglement, text-image interaction, automatic speech recognition, text-to-speech synthesis, seed sensitivity, and cross-modal paradigm comparison.

### B. Consolidated Answers to the Research Questions

**RQ1 (Uniform scale):** The IP-Adapter scale $\lambda$ produces a nonlinear, monotonic transition from unconditioned generation to reference-guided synthesis, with the most perceptually interesting variation in the $[0.4, 0.7]$ range. The relationship is smooth and generalizes across reference images.

**RQ2 (Block-level configurations):** Different block-level configurations produce meaningfully different style transfer outcomes. Deeper blocks transfer global stylistic attributes, multi-layer activation produces additive effects, and the content-style disentanglement is robust across diverse text prompts and reference images.

**RQ3 (Text-image interaction):** Text and image conditioning interact additively. The transition from text-dominated to image-dominated output is gradual. Prompt specificity modulates the effective text contribution, and semantic compatibility between text and image improves output coherence.

**RQ4 (ASR models and preprocessing):** Whisper-medium achieved the lowest WER (0.172) on MINDS-14, outperforming wav2vec 2.0 (0.407 WER) at the cost of 25× longer inference time. Sampling rate mismatch produced near-catastrophic WER degradation for both architectures.

**RQ5 (Bark TTS):** Bark produces substantial prosodic variation across seeds (duration std: 21% of mean) with consistent semantic content. Generation time scales linearly with text length at ~2.3× real-time. Round-trip TTS→ASR evaluation confirms general intelligibility.

**RQ6 (Seed sensitivity):** Seed sensitivity varies dramatically across paradigms — very low for IP-Adapter image generation (especially block-level) and high for Bark TTS, reflecting the fundamental architectural difference between conditioned diffusion and multi-stage autoregressive sampling.

**RQ7 (Cross-modal trade-offs):** IP-Adapter provides rich, continuous, fine-grained control with moderate cost. ASR is fast and deterministic with well-defined metrics. Bark TTS achieves remarkable text-to-audio transformation with high stochasticity and computational cost. The controllability-diversity trade-off appears fundamental to the diffusion-versus-autoregressive distinction.

### C. Final Takeaways for IP-Adapter, ASR, and Bark-Based Generation

The IP-Adapter framework demonstrates that powerful conditioning modalities can be added to frozen diffusion models via lightweight adapters with decoupled cross-attention — a paradigm of efficient, modular generative control. The block-level scaling mechanism extends this to structured conditioning, enabling content-style disentanglement that uniform scaling cannot achieve. For ASR deployment, model selection and correct preprocessing are both critical, with Whisper offering the best accuracy-cost trade-off for general-purpose applications. Bark TTS, while computationally expensive and stochastically variable, achieves naturalistic speech synthesis from text through an elegant three-stage autoregressive cascade over discrete audio tokens.

---

<div style="page-break-after: always;"></div>

## XIII. Future Work

### A. Vision Extensions

Future work in the vision domain could extend this study along several axes: (1) computing pixel-level similarity metrics (SSIM, LPIPS) and perceptual quality scores (FID) to complement the qualitative visual analysis with quantitative reference-fidelity curves; (2) exploring the full combinatorial space of multi-block configurations — simultaneously activating layers across multiple upsampling blocks — to characterize synergistic and antagonistic block interactions; (3) investigating IP-Adapter behavior with SDXL's refiner model or alternative base models to test the generality of the scale-fidelity relationships observed here.

### B. Audio Extensions

In the audio domain, future work could: (1) expand the ASR evaluation to the full MINDS-14 multilingual subset to characterize cross-lingual robustness of the model ranking observed on en-AU; (2) compare `bark-small` against the full `suno/bark` model to quantify the quality-efficiency trade-off of the smaller variant; (3) investigate Bark's speaker embedding and voice preset capabilities for controlled speaker variation; (4) apply more sophisticated TTS evaluation metrics (MOS estimation, spectral analysis) to complement the round-trip WER approach.

### C. Cross-Modal Extensions

At the cross-modal level, future work could: (1) design a unified framework for comparing controllability across generative paradigms using information-theoretic measures; (2) investigate hybrid pipelines that chain Bark TTS output with diffusion-based audio-to-image systems to create end-to-end text-to-image pipelines mediated by audio representations; (3) conduct a systematic comparison of adapter-based conditioning (IP-Adapter) with other efficient adaptation methods (LoRA, textual inversion) across both vision and audio domains.

---

<div style="page-break-after: always;"></div>

## XIV. References

[1] Alammar, J. & Grootendorst, M. *Hands-on Generative AI with Transformers and Diffusion Models.* O'Reilly Media, 2024.

[2] Ye, H., Zhang, J., Liu, S., Han, X., & Yang, W. "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models." *arXiv preprint arXiv:2308.06721*, 2023.

[3] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller, J., Penna, J., & Rombach, R. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." *arXiv preprint arXiv:2307.01952*, 2023.

[4] Gerz, D., Vulic, I., Ponti, E.M., Buber, J., Mrksic, N., Coope, S., Razavi, A., Steedman, M., & Henderson, M. "Multilingual and Cross-Lingual Intent Detection from Spoken Data." *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 4698-4713, 2021.

[5] Suno AI. "Bark: Text-Prompted Generative Audio Model." GitHub repository, 2023.

[6] Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models." *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 6840-6851, 2020.

[7] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 10684-10695, 2022.

[8] Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234-241, 2015.

[9] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. "Learning Transferable Visual Models From Natural Language Supervision." *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 8748-8763, 2021.

[10] Ho, J. & Salimans, T. "Classifier-Free Diffusion Guidance." *arXiv preprint arXiv:2207.12598*, 2022.

[11] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 12449-12460, 2020.

[12] Graves, A., Fernandez, S., Gomez, F., & Schmidhuber, J. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks." *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 369-376, 2006.

[13] Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. "Robust Speech Recognition via Large-Scale Weak Supervision." *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 28492-28518, 2023.

[14] Defossez, A., Copet, J., Synnaeve, G., & Adi, Y. "High Fidelity Neural Audio Compression." *Transactions on Machine Learning Research (TMLR)*, 2023.

---

**Author:** Ryan Kamp  
**Affiliation:** CS6078 Generative AI, University of Cincinnati  
**Date:** March 2026