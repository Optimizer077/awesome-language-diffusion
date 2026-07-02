# Awesome Language Diffusion [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)
![Papers](https://img.shields.io/badge/papers-311-blue)
![Topics](https://img.shields.io/badge/topics-17-orange)
![Last updated](https://img.shields.io/badge/updated-2026--07-informational)

> A curated, aggressively-sourced list of papers, models, code, and resources on **diffusion models for language** — text, code, and discrete-token sequences.
>
> **311 verified references** spanning foundations → discrete/masked theory → 7B–100B diffusion LLMs → multimodal/VLA, reasoning, coding, RL, inference, safety, and evaluation.

> 📊 **[PAPERS.md](PAPERS.md) — full sortable master index of all 311 papers with years**, newest-first. Read the topical sections below to learn; scan `PAPERS.md` to browse everything by date.

Language diffusion (a.k.a. **diffusion language models, dLLMs, discrete/masked diffusion LMs**) generates text by *iteratively denoising* a whole sequence in parallel instead of predicting one token left-to-right. In 2024–2026 this went from a research curiosity to **8B–100B open models (LLaDA, Dream, LLaDA-MoE/2.0)** and **frontier products (Mercury, Gemini Diffusion, Seed Diffusion)** that reach **1000–2000+ tokens/s**.

This list is organized so you can read it as a *learning path* (foundations → models → frontiers) **and** use it as a *reference* (pick a subtopic table).

<p align="center"><i>Contributions welcome — see <a href="CONTRIBUTING.md">CONTRIBUTING.md</a>. Found a broken link or wrong attribution? Open an issue.</i></p>

---

## Contents

- [Why diffusion for language?](#why-diffusion-for-language)
- [Models at a glance](#models-at-a-glance)
- [Surveys & overviews](#surveys--overviews)
- [1. Foundations & theory](#1-foundations--theory)
  - [1.1 Core diffusion (continuous)](#11-core-diffusion-continuous)
  - [1.2 Discrete diffusion theory](#12-discrete-diffusion-theory)
  - [1.3 Masked / absorbing diffusion (modern)](#13-masked--absorbing-diffusion-modern)
  - [1.4 Advanced theory: samplers, correctors, distillation & likelihood](#14-advanced-theory-samplers-correctors-distillation--likelihood)
- [2. Early & continuous text diffusion](#2-early--continuous-text-diffusion)
  - [2.1 Continuous / embedding diffusion LMs](#21-continuous--embedding-diffusion-lms)
  - [2.1b Discrete & any-order text diffusion (pre-dLLM)](#21b-discrete--any-order-text-diffusion-pre-dllm)
  - [2.2 Latent-space text diffusion](#22-latent-space-text-diffusion)
  - [2.3 Iterative-refinement roots](#23-iterative-refinement-roots)
- [3. Large-scale diffusion LLMs (dLLMs)](#3-large-scale-diffusion-llms-dllms)
  - [3.1 Open diffusion LLMs](#31-open-diffusion-llms)
  - [3.2 Frontier / commercial dLLMs](#32-frontier--commercial-dllms)
  - [3.3 Hybrid & block diffusion (AR ↔ diffusion)](#33-hybrid--block-diffusion-ar--diffusion)
- [4. Scaling laws & training](#4-scaling-laws--training)
- [5. Reasoning](#5-reasoning)
- [6. Coding & agents](#6-coding--agents)
- [7. RL, post-training & alignment](#7-rl-post-training--alignment)
- [8. Multimodal & unified diffusion](#8-multimodal--unified-diffusion)
- [9. Inference & acceleration](#9-inference--acceleration)
- [10. Guidance, control & constrained decoding](#10-guidance-control--constrained-decoding)
- [11. Applications](#11-applications)
- [12. Safety, watermarking & robustness](#12-safety-watermarking--robustness)
- [13. Efficiency: quantization](#13-efficiency-quantization)
- [14. Embeddings, energy-based & Bayesian flow networks](#14-embeddings-energy-based--bayesian-flow-networks)
- [15. "Language of X": biology & speech](#15-language-of-x-biology--speech)
- [16. Evaluation & benchmarks](#16-evaluation--benchmarks)
- [17. Extended index — deep gap-sweep (2022–2026)](#17-extended-index--deep-gap-sweep-20222026)
- [📊 PAPERS.md — full sortable master index (311 papers, with years)](PAPERS.md)
- [Resources](#resources)
  - [Code & checkpoints](#code--checkpoints)
  - [Libraries & frameworks](#libraries--frameworks)
  - [Datasets & benchmarks](#datasets--benchmarks)
  - [Blogs & talks](#blogs--talks)
  - [Demos](#demos)
- [Recent trends (2024–2026)](#recent-trends-20242026)
- [Contributing](#contributing)
- [License](#license)

---

## Why diffusion for language?

| | Autoregressive LM | Diffusion LM |
|---|---|---|
| **Decoding** | one token, left→right | many tokens in parallel, coarse→fine |
| **Context** | causal (past only) | bidirectional (past + future) |
| **Speed** | O(n) sequential steps | O(k) denoising steps, k ≪ n possible |
| **Native abilities** | streaming, KV-cache | infilling, any-order generation, global control |
| **Main challenge** | — | likelihood/perplexity gap, sampling quality, KV-cache |

**Two families:** *continuous* diffusion adds Gaussian noise to token **embeddings** (Diffusion-LM, Plaid); *discrete/masked* diffusion corrupts tokens toward a `[MASK]`/uniform state and learns to un-mask (D3PM, SEDD, MDLM) — the discrete/masked branch dominates modern dLLMs.

> **A note on very recent preprints.** This list intentionally includes bleeding-edge 2025–2026 arXiv papers. Entries dated late-2025/2026 are fast-moving; when in doubt, confirm the arXiv ID and claims at the source before citing.

---

## Models at a glance

Notable diffusion language models you can actually run or query, newest-scale first. "Type" = M (masked/discrete), B (block/hybrid), C (continuous).

| Model | Params | Type | Open? | Speciality | Links |
|---|---|---|---|---|---|
| **LLaDA 2.0** | 100B / 16B MoE | M+B | ✅ weights | Largest open dLLM | [abs](https://arxiv.org/abs/2512.15745) |
| **RND1** | 30B (3B active) MoE | M | ✅ weights | AR→diffusion conversion | [report](https://www.radicalnumerics.ai/assets/rnd1_report.pdf) |
| **LLaDA-MoE** | 7B (1.4B active) | M | ✅ weights | First from-scratch MoE dLLM | [abs](https://arxiv.org/abs/2509.24389) |
| **LLaDA / 1.5** | 8B | M | ✅ weights | Reference open dLLM | [abs](https://arxiv.org/abs/2502.09992) · [HF](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) |
| **Dream 7B** | 7B | M | ✅ weights | Strong planning, any-order | [abs](https://arxiv.org/abs/2508.15487) · [code](https://github.com/DreamLM/Dream) |
| **DiffuCoder** | 7B | M | ✅ weights | Code + coupled-GRPO | [HF](https://huggingface.co/apple/DiffuCoder-7B-Instruct) |
| **Dream-Coder** | 7B | M | ✅ weights | Open code dLLM | [abs](https://arxiv.org/abs/2509.01142) |
| **LLaDA-V / MMaDA** | 8B | M | ✅ weights | Multimodal / unified | [abs](https://arxiv.org/abs/2505.16933) · [abs](https://arxiv.org/abs/2505.15809) |
| **Mercury** | — | M/B | 🔒 API | Fastest commercial (1000+ tok/s) | [demo](https://chat.inceptionlabs.ai) |
| **Gemini Diffusion** | — | — | 🔒 waitlist | Frontier-lab dLLM | [page](https://deepmind.google/models/gemini-diffusion/) |
| **Seed Diffusion** | — | B | 🔒 | Code, 2146 tok/s | [abs](https://arxiv.org/abs/2508.02193) |
| **TESS 2** | — | C | ✅ weights | Continuous generalist | [abs](https://arxiv.org/abs/2502.13917) |
| **SEDD / MDLM / MD4** | ≤ GPT-2 | M | ✅ weights | Research baselines | [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) · [MDLM](https://github.com/kuleshov-group/mdlm) |

---

## Surveys & overviews

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **A Survey on Diffusion Language Models** | arXiv'25 | [abs](https://arxiv.org/abs/2508.10875) · [list](https://github.com/VILA-Lab/Awesome-DLMs) | Comprehensive taxonomy of continuous/discrete/multimodal DLMs; the reference survey + companion Awesome-DLMs. |
| **Discrete Diffusion in Large Language & Multimodal Models: A Survey** | arXiv'25 | [abs](https://arxiv.org/abs/2506.13759) | Systematic survey of dLLMs and discrete-diffusion multimodal models: parallel decoding, control, ~10× speedups. |
| **A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion LMs** | arXiv'25 | [abs](https://arxiv.org/abs/2508.08712) | Frames dLLMs within the broader parallel-generation landscape. |
| **Promises, Outlooks and Challenges of Diffusion Language Modeling** | arXiv'24 | [abs](https://arxiv.org/abs/2406.11473) | Careful eval of SEDD vs GPT-2 on perplexity, benchmarks, latency; honest look at limits. |
| **Diffusion Models for Non-autoregressive Text Generation: A Survey** | IJCAI'23 | [abs](https://arxiv.org/abs/2303.06574) · [list](https://github.com/RUCAIBox/Awesome-Text-Diffusion-Models) | First dedicated survey of text diffusion. |
| **A Survey of Diffusion Models in Natural Language Processing** | arXiv'23 | [abs](https://arxiv.org/abs/2305.14671) | Broad early NLP-diffusion survey. |

---

## 1. Foundations & theory

### 1.1 Core diffusion (continuous)

The general diffusion machinery that text diffusion inherits.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** | ICML'15 | [abs](https://arxiv.org/abs/1503.03585) | Origin of the forward-noising / learned-reverse framework. |
| **Generative Modeling by Estimating Gradients (NCSN)** | NeurIPS'19 | [abs](https://arxiv.org/abs/1907.05600) | Score matching + annealed Langevin: the score-based root. |
| **Denoising Diffusion Probabilistic Models (DDPM)** | NeurIPS'20 | [abs](https://arxiv.org/abs/2006.11239) | The canonical DDPM with the simple noise-prediction objective. |
| **Denoising Diffusion Implicit Models (DDIM)** | ICLR'21 | [abs](https://arxiv.org/abs/2010.02502) | Non-Markovian deterministic sampler, 10–50× faster. |
| **Score-Based Generative Modeling through SDEs** | ICLR'21 | [abs](https://arxiv.org/abs/2011.13456) | Unifies score & diffusion under a continuous reverse-time SDE. |
| **Improved DDPM** | ICML'21 | [abs](https://arxiv.org/abs/2102.09672) | Learned variances + cosine schedule improve likelihood. |
| **Diffusion Models Beat GANs (classifier guidance)** | NeurIPS'21 | [abs](https://arxiv.org/abs/2105.05233) | Introduces classifier guidance. |
| **Classifier-Free Diffusion Guidance** | arXiv'22 | [abs](https://arxiv.org/abs/2207.12598) | The now-default conditioning trick, reused for text. |
| **Flow Matching for Generative Modeling** | ICLR'23 | [abs](https://arxiv.org/abs/2210.02747) | Simulation-free CNF training generalizing diffusion objectives. |

### 1.2 Discrete diffusion theory

Extending diffusion to categorical / token data.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Argmax Flows & Multinomial Diffusion** | NeurIPS'21 | [abs](https://arxiv.org/abs/2102.05379) | First categorical/multinomial diffusion for discrete data. |
| **D3PM: Structured Denoising Diffusion in Discrete State-Spaces** | NeurIPS'21 | [abs](https://arxiv.org/abs/2107.03006) | Generalizes discrete diffusion via transition matrices; introduces **absorbing (mask)** corruption. |
| **Autoregressive Diffusion Models (ARDM)** | ICLR'22 | [abs](https://arxiv.org/abs/2110.02037) | Any-order autoregressive/diffusion bridge for discrete data — conceptual root of any-order dLLMs. |
| **A Continuous-Time Framework for Discrete Denoising Models** | NeurIPS'22 | [abs](https://arxiv.org/abs/2205.14987) | First full continuous-time (CTMC) treatment + ELBO. |
| **Concrete Score Matching** | NeurIPS'22 | [abs](https://arxiv.org/abs/2211.00802) | Defines the "concrete score" generalizing score matching to discrete data — basis of SEDD. |
| **Generative Flows on Discrete State-Spaces (Multiflow)** | ICML'24 | [abs](https://arxiv.org/abs/2402.04997) | Multimodal discrete flow framework (text + protein). |

### 1.3 Masked / absorbing diffusion (modern)

The theory that made discrete diffusion competitive with AR.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **SEDD: Discrete Diffusion by Estimating Ratios** | ICML'24 🏆 | [abs](https://arxiv.org/abs/2310.16834) · [code](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | **Score entropy** loss; first to beat GPT-2-scale AR in generative perplexity. Revived the field. |
| **MDLM: Simple & Effective Masked Diffusion LMs** | NeurIPS'24 | [abs](https://arxiv.org/abs/2406.07524) · [code](https://github.com/kuleshov-group/mdlm) | SUBS parameterization + Rao-Blackwellized bound; standard masked-diffusion baseline. |
| **MD4: Simplified & Generalized Masked Diffusion** | NeurIPS'24 | [abs](https://arxiv.org/abs/2406.04329) · [code](https://github.com/google-deepmind/md4) | Reduces the objective to a weighted integral of cross-entropy losses. |
| **RADD: Your Absorbing Discrete Diffusion Secretly Models…** | ICLR'25 | [abs](https://arxiv.org/abs/2406.03736) · [code](https://github.com/ML-GSAI/RADD) | Absorbing-diffusion score is time-independent clean-data conditionals; unifies with any-order AR. |
| **Discrete Flow Matching** | NeurIPS'24 | [abs](https://arxiv.org/abs/2407.15595) | Flow matching for discrete sequences with general probability paths. |
| **Masked Diffusion Models are Secretly Time-Agnostic Masked Models** | arXiv'24 | [abs](https://arxiv.org/abs/2409.02908) | Reveals time-agnosticity + sampling flaws in masked diffusion; sharper theory. |
| **GIDD: Generalized Interpolating Discrete Diffusion** | arXiv'25 | [abs](https://arxiv.org/abs/2503.04482) | Unifies masking and uniform-noise diffusion; enables self-correction. |
| **The Diffusion Duality (Duo)** | ICML'25 | [abs](https://arxiv.org/abs/2506.10892) | Links uniform-state discrete diffusion to Gaussian diffusion → curriculum training + consistency distillation. |

### 1.4 Advanced theory: samplers, correctors, distillation & likelihood

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Informed Correctors for Discrete Diffusion** | arXiv'24 | [abs](https://arxiv.org/abs/2407.21243) | Model-informed predictor-corrector sampler + hollow transformer; sharply cuts few-step error. |
| **DDPD: Think While You Generate (Planned Denoising)** | ICLR'25 | [abs](https://arxiv.org/abs/2410.06264) | Splits generation into a planner (which position to fix) + denoiser; Gillespie-based adaptive sampler. |
| **Train for the Worst, Plan for the Best (token ordering)** | ICML'25 🏆 | [abs](https://arxiv.org/abs/2502.06768) | Theory of MDM train/inference tradeoff: adaptive any-order inference can beat fixed-order AR on planning. |
| **Any-Order GPT as Masked Diffusion Model** | arXiv'25 | [abs](https://arxiv.org/abs/2506.19935) | Decouples the any-order-AR objective from architecture; clarifies AO-ARM ↔ masked-diffusion equivalence. |
| **TCSM: Target Concrete Score Matching** | ICML'25 | [abs](https://arxiv.org/abs/2504.16431) | Unifies training, distillation & reward fine-tuning under one target-concrete-score objective. |
| **Di4C: Distillation through Dimensional Correlations** | ICML'25 | [abs](https://arxiv.org/abs/2410.08709) | Few-step distillation capturing inter-dimensional correlations lost by factorized denoisers. |
| **Fast Sampling via Non-Markov Discrete Diffusion** | NeurIPS'24 | [abs](https://arxiv.org/abs/2312.09193) | Predetermined transition times enable faster, higher-quality sampling than Markov chains. |
| **CEDD: Efficient Perplexity Bound & Ratio Matching** | arXiv'25 | [abs](https://arxiv.org/abs/2507.04341) | Tightens the ELBO/perplexity bound; cross-entropy training beats direct ratio matching (SEDD family). |
| **Convergence of Score-Based Discrete Diffusion** | arXiv'25 | [abs](https://arxiv.org/abs/2410.02321) | Discrete-time KL convergence bounds (score/discretization/truncation) for CTMC discrete diffusion. |
| **Discrete State Diffusion: A Sample-Complexity Perspective** | arXiv'25 | [abs](https://arxiv.org/abs/2510.10854) | Learning-theoretic sample-complexity bounds complementing convergence analyses. |
| **DUEL: Exact Likelihood via Deterministic Unmasking** | arXiv'26 | [abs](https://arxiv.org/abs/2603.01367) | Derives exact (non-ELBO) any-order likelihood for masked diffusion. |
| **Variational Masked Diffusion Models** | arXiv'25 | [abs](https://arxiv.org/abs/2510.23606) | Adds latent variables to capture inter-token dependencies factorized denoisers miss. |
| **Latent-Augmented Discrete Diffusion** | arXiv'25 | [abs](https://arxiv.org/abs/2510.18114) | Continuous latents feed soft probabilities to the denoiser, lowering training variance. |
| **LatentLM: Multimodal Latent LM w/ Next-Token Diffusion** | arXiv'24 | [abs](https://arxiv.org/abs/2412.08635) | Continuous-latent alternative: generate VAE latents via next-token diffusion; unifies discrete+continuous. |
| **Edit Flows: Flow Matching with Edit Operations** | arXiv'25 | [abs](https://arxiv.org/abs/2506.09018) | CTMC over variable-length sequences via insert/delete/substitute — insertion-deletion alternative to masking. |

---

## 2. Early & continuous text diffusion

### 2.1 Continuous / embedding diffusion LMs

Diffuse over word **embeddings** or the **vocabulary simplex** (mostly 2022–2023).

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Diffusion-LM Improves Controllable Text Generation** | NeurIPS'22 | [abs](https://arxiv.org/abs/2205.14217) · [code](https://github.com/XiangLi1999/Diffusion-LM) | Seminal continuous word-embedding diffusion LM; gradient-based fine-grained control. |
| **DiffuSeq: Seq2Seq Text Generation with Diffusion** | ICLR'23 | [abs](https://arxiv.org/abs/2210.08933) · [code](https://github.com/Shark-NLP/DiffuSeq) | First classifier-free seq2seq embedding diffusion; high-diversity conditional text. |
| **SSD-LM: Semi-autoregressive Simplex Diffusion** | ACL'23 | [abs](https://arxiv.org/abs/2210.17432) | Diffuses over the vocab simplex in blocks; plug-and-play control with off-the-shelf classifiers. |
| **CDCD: Continuous Diffusion for Categorical Data** | arXiv'22 | [abs](https://arxiv.org/abs/2211.15089) | Score interpolation + time warping; a widely-cited continuous backbone. |
| **SED: Self-conditioned Embedding Diffusion** | arXiv'22 | [abs](https://arxiv.org/abs/2211.04236) | Scalable continuous diffusion on token embeddings with self-conditioning. |
| **Difformer: Diffusion on the Embedding Space** | NAACL'24 | [abs](https://arxiv.org/abs/2212.09412) | Diagnoses & fixes embedding-diffusion failure modes (anchor loss, embedding LayerNorm). |
| **SeqDiffuSeq: Encoder-Decoder Text Diffusion** | NAACL'24 | [abs](https://arxiv.org/abs/2212.10325) | Encoder-decoder + self-conditioning + token-level adaptive noise schedule. |
| **GENIE: Large-Scale Pre-training for Text Diffusion** | ICML'23 | [abs](https://arxiv.org/abs/2212.11685) | First large-scale pre-trained diffusion LM; continuous paragraph denoise objective. |
| **DINOISER: Manipulating Noises for Conditional Seq Learning** | TACL'23 | [abs](https://arxiv.org/abs/2302.10025) | Noise-scale is the key pitfall; manipulate it to boost MT & conditioning. |
| **TESS: Text-to-Text Self-Conditioned Simplex Diffusion** | EACL'24 | [abs](https://arxiv.org/abs/2305.08379) | Fully NAR diffusion on the logit-simplex; competitive with pretrained seq2seq. |
| **AR-Diffusion: Auto-Regressive Diffusion for Text** | NeurIPS'23 | [abs](https://arxiv.org/abs/2305.09515) | Position-dependent denoising injects left→right bias into diffusion. |
| **Plaid: Likelihood-Based Diffusion Language Models** | NeurIPS'23 | [abs](https://arxiv.org/abs/2305.18619) · [code](https://github.com/igul222/plaid) | Closes the likelihood gap; releases Plaid 1B; studies diffusion-LM scaling. |
| **SSD-2: David helps Goliath (13B, inference collaboration)** | NAACL'24 | [abs](https://arxiv.org/abs/2305.14771) | Scales SSD-LM to 13B; diffusion LMs ensemble well at inference. |
| **DiffuSeq-v2: Bridging Discrete & Continuous** | EMNLP'23 | [abs](https://arxiv.org/abs/2310.05793) | Soft absorbing state + ODE solvers → 4× faster train, 800× faster sampling. |

### 2.1b Discrete & any-order text diffusion (pre-dLLM)

Discrete/masked diffusion for text before the large-scale dLLM era.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **DiffusionBERT: Generative Masked LMs with Diffusion** | ACL'23 | [abs](https://arxiv.org/abs/2211.15029) | Marries BERT with absorbing diffusion + spike noise schedule. |
| **RDM: A Reparameterized Discrete Diffusion Model for Text** | arXiv'23 | [abs](https://arxiv.org/abs/2302.05737) | Reparameterization enabling adaptive routing/decoding. |
| **Masked-Diffuse LM: Cheaper & Better with Soft-Masked Noise** | EMNLP'23 | [abs](https://arxiv.org/abs/2304.04746) | Linguistically-informed soft-masking noise schedule. |
| **Diffusion-NAT: Self-Prompting Discrete Diffusion (BART)** | EACL'24 | [abs](https://arxiv.org/abs/2305.04044) | Unifies BART decoding with discrete diffusion. |
| **Diffusion LMs Can Perform Many Tasks with Scaling & Instruction-Finetuning** | arXiv'23 | [abs](https://arxiv.org/abs/2308.12219) | Instruction-tuned diffusion LMs as general task solvers. |
| **FiLM: Fill-in Language Models for Any-Order Generation** | arXiv'23 | [abs](https://arxiv.org/abs/2310.09930) | Any-order infilling bridging masked LM and diffusion. |

### 2.2 Latent-space text diffusion

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **LD4LG: Latent Diffusion for Language Generation** | NeurIPS'23 | [abs](https://arxiv.org/abs/2212.09462) | Diffusion in a pretrained encoder-decoder's latent space; strong with few steps. |

### 2.3 Iterative-refinement roots

Pre-diffusion NAR ideas whose techniques (self-conditioning, unrolled denoising) power text diffusion.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **SUNDAE: Step-unrolled Denoising Autoencoders** | ICLR'22 | [abs](https://arxiv.org/abs/2112.06749) | Non-AR iterative refinement converging in few steps. |
| **Analog Bits / Bit Diffusion** | ICLR'23 | [abs](https://arxiv.org/abs/2208.04202) | Discrete tokens as continuous "analog bits" + self-conditioning (reused everywhere). |

---

## 3. Large-scale diffusion LLMs (dLLMs)

The 2025 breakthrough: masked diffusion at 7B–100B, competitive with AR LLMs.

### 3.1 Open diffusion LLMs

| Model | Venue | Links | TL;DR |
|---|---|---|---|
| **LLaDA — Large Language Diffusion Models** | arXiv'25 | [abs](https://arxiv.org/abs/2502.09992) · [code](https://github.com/ML-GSAI/LLaDA) · [demo](https://huggingface.co/spaces/multimodalart/LLaDA) | First **8B** masked-diffusion LLM from scratch rivaling LLaMA3-8B; the field's reference model. |
| **Dream 7B** | arXiv'25 | [abs](https://arxiv.org/abs/2508.15487) · [code](https://github.com/DreamLM/Dream) · [blog](https://hkunlp.github.io/blog/2025/dream/) | Strongest fully-open 7B dLLM at release; AR-init + context-adaptive noise; strong planning. |
| **LLaDA-MoE** | arXiv'25 | [abs](https://arxiv.org/abs/2509.24389) | First from-scratch **MoE** dLLM (7B total / ~1.4B active, ~20T tokens). |
| **LLaDA 2.0 — Scaling to 100B** | arXiv'25 | [abs](https://arxiv.org/abs/2512.15745) | Scales dLLMs to **100B** MoE via AR→dLLM conversion + 3-phase block-diffusion training. |
| **DiffuLLaMA / DiffuGPT — Scaling via Adaptation from AR** | ICLR'25 | [abs](https://arxiv.org/abs/2410.17891) · [code](https://github.com/HKUNLP/DiffuLLaMA) | Converts GPT2/LLaMA (127M–7B) into diffusion LMs with <200B tokens — cheap recipe. |
| **RND1 (30B/3B-active MoE, from Qwen3)** | report'25 | [report](https://www.radicalnumerics.ai/assets/rnd1_report.pdf) · [HF](https://huggingface.co/radicalnumerics) | Largest open dLLM at release; open AR→diffusion (A2D) recipe. |
| **TESS 2 — Generalist Diffusion LM** | ACL'25 | [abs](https://arxiv.org/abs/2502.13917) | Instruction-following generalist dLLM via AR adaptation + inference-time reward guidance. |
| **OpenMoE 2: Sparse Diffusion Language Models** | project'25 | [code](https://github.com/JinjieNi/OpenMoE2) | From-scratch study of expert-choice MoE × diffusion LMs with open checkpoints/logs. |
| **TraDo-8B (TraceRL)** | arXiv'25 | [abs](https://arxiv.org/abs/2509.06949) | First long-CoT diffusion LLM, trained with trajectory-aware RL on block diffusion (see §7). |

### 3.2 Frontier / commercial dLLMs

| Model | Org | Links | TL;DR |
|---|---|---|---|
| **Mercury — Ultra-Fast Diffusion LMs** | Inception Labs | [abs](https://arxiv.org/abs/2506.17298) · [blog](https://www.inceptionlabs.ai/blog/introducing-mercury) · [demo](https://chat.inceptionlabs.ai) | First commercial-scale dLLM; Mercury Coder **1000+ tok/s** on H100 at frontier quality. |
| **Gemini Diffusion** | Google DeepMind | [page](https://deepmind.google/models/gemini-diffusion/) · [blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/) | Frontier-lab text diffusion (1000–2000 tok/s) matching Gemini 2.0 Flash-Lite on coding. |
| **Seed Diffusion** | ByteDance Seed | [abs](https://arxiv.org/abs/2508.02193) | Large-scale code dLLM at **2146 tok/s** (5.4× faster than comparable AR). |

### 3.3 Hybrid & block diffusion (AR ↔ diffusion)

Interpolating between autoregression and diffusion to recover KV-cache & quality.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **BD3-LM: Block Diffusion** | ICLR'25 (Oral) | [abs](https://arxiv.org/abs/2503.09573) · [code](https://github.com/kuleshov-group/bd3lms) | Block-wise diffusion → KV-caching, variable length, tunable AR↔diffusion; SOTA among diffusion LMs. |
| **Eso-LMs: Esoteric (any-order) Language Models** | arXiv'25 | [abs](https://arxiv.org/abs/2506.01928) | Fuses AR + masked diffusion; first to add KV caching to MDMs (up to 65× faster). |
| **SDLM: Sequential Diffusion Language Models** | arXiv'25 | [abs](https://arxiv.org/abs/2509.24007) | Next-Sequence-Prediction retrofits AR models into adaptive-length, KV-compatible block diffusion. |
| **SDAR: Synergistic Diffusion-AutoRegression** | arXiv'25 | [abs](https://arxiv.org/abs/2510.06303) | Cheap AR→block-diffusion conversion (1.7B–30B) preserving AR quality, parallel block decoding. |
| **TiDAR: Think in Diffusion, Talk in Autoregression** | arXiv'25 | [abs](https://arxiv.org/abs/2511.08923) | Single-pass hybrid: drafts in diffusion, samples in AR; closes AR quality gap at 4.7–5.9× throughput. |

---

## 4. Scaling laws & training

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **SMDM: Scaling up Masked Diffusion on Text** | ICLR'25 | [abs](https://arxiv.org/abs/2410.18514) | First scaling law for masked diffusion; AR-comparable scaling rate up to 1.1B. |
| **Quokka: Training Optimal Large Diffusion LMs** | arXiv'25 | [abs](https://arxiv.org/abs/2510.03280) | First compute- & data-constrained scaling law for dLLMs ("Chinchilla for diffusion"). |
| **Diffusion Beats Autoregressive in Data-Constrained Settings** | NeurIPS'25 | [abs](https://arxiv.org/abs/2507.15857) | Masked diffusion wins when **data** (not compute) is the bottleneck — implicit augmentation from token orderings. |
| **Diffusion Language Models are Super Data Learners** | arXiv'25 | [abs](https://arxiv.org/abs/2511.03276) | dLLMs overtake AR under limited unique data ("crossover") via any-order + built-in MC augmentation. |
| **Scaling Behavior of Discrete Diffusion Language Models** | arXiv'25 | [abs](https://arxiv.org/abs/2512.10858) | Masked vs uniform diffusion up to 10B: **uniform** more data-efficient in data-bound regimes. |
| **Scaling Beyond Masked Diffusion Language Models** | ICML'26 | [abs](https://arxiv.org/abs/2602.15014) | IsoFLOP study: masked diffusion isn't optimal; uniform/interpolating models scale better. |

---

## 5. Reasoning

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Diffusion of Thoughts (DoT)** | arXiv'24 | [abs](https://arxiv.org/abs/2402.07754) | Chain-of-thought that "diffuses" over time; self-correction + compute-accuracy tradeoff. |
| **Beyond Autoregression: Discrete Diffusion for Reasoning & Planning (MGDM)** | ICLR'25 | [abs](https://arxiv.org/abs/2410.14157) | Masters hard subgoals AR misses: 91.5% / 100% on Countdown / Sudoku without search. |
| **RFG: Reward-Free Guidance test-time scaling** | arXiv'25 | [abs](https://arxiv.org/abs/2509.25604) | Guides any-order dLLM reasoning via log-likelihood ratios of RL/SFT vs reference — no process reward. |

*(See also RL-based reasoning: [d1](https://arxiv.org/abs/2504.12216), [wd1](https://arxiv.org/abs/2507.08838), [GDPO](https://arxiv.org/abs/2510.08554), [DiFFPO](https://arxiv.org/abs/2510.02212), [DMPO](https://arxiv.org/abs/2510.08233) in §7.)*

---

## 6. Coding & agents

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **CodeFusion** | EMNLP'23 (Oral) | [abs](https://arxiv.org/abs/2310.17680) | First pre-trained diffusion code model; 75M rivals much larger AR on top-k accuracy. |
| **DiffuCoder** | arXiv'25 | [abs](https://arxiv.org/abs/2506.20639) · [code](https://github.com/apple/ml-diffucoder) | Apple's 7B code dLLM + coupled-GRPO RL; analyzes dLLM decoding order vs AR (+4.4% EvalPlus). |
| **Dream-Coder 7B** | arXiv'25 | [abs](https://arxiv.org/abs/2509.01142) · [blog](https://hkunlp.github.io/blog/2025/dream-coder/) | Fully-open code dLLM; emergent any-order decoding + verifiable-reward RL. |
| **CoDA: Coding LM via Diffusion Adaptation** | arXiv'25 | [abs](https://arxiv.org/abs/2510.03270) | Tiny 1.7B code dLLM (from Qwen3-1.7B) matching up to 7B diffusion models. |
| **Beyond Autoregression: Empirical Study of dLLMs for Code** | arXiv'25 | [abs](https://arxiv.org/abs/2509.11252) | Quantifies multi-token + flexible-order advantages for non-sequential code editing. |
| **The Bitter Lesson of Diffusion LMs for Agentic Workflows** | ACL'26 | [abs](https://arxiv.org/abs/2601.12979) | Reality check (DiffuAgent framework): current dLLMs fail at causal planning / strict-JSON tool calls, help in non-causal roles. |
| **DLLM-Searcher: dLLMs as Search Agents** | arXiv'26 | [abs](https://arxiv.org/abs/2602.07035) | Parallel reason-and-act paradigm turning dLLMs into retrieval/tool-using agents. |

*(See also [Dream-VL & Dream-VLA](#vision-language-action-robotics) under Vision-Language-Action.)*

---

## 7. RL, post-training & alignment

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **d1: Scaling Reasoning via RL (diffu-GRPO)** | NeurIPS'25 | [abs](https://arxiv.org/abs/2504.12216) | First policy-gradient RL for masked dLLMs + masked SFT; top GSM8K/MATH among dLLMs. |
| **LLaDA 1.5: Variance-Reduced Preference Optimization (VRPO)** | arXiv'25 | [abs](https://arxiv.org/abs/2505.19223) · [HF](https://huggingface.co/GSAI-ML/LLaDA-1.5) | Tames high-variance ELBO gradients → stable DPO-style alignment for masked diffusion. |
| **wd1: Weighted Policy Optimization** | arXiv'25 | [abs](https://arxiv.org/abs/2507.08838) | Ratio-free weighted-log-likelihood RL; beats d1 on LLaDA-8B by up to +59% at lower cost. |
| **GDPO: Group Diffusion Policy/Preference Optimization** | arXiv'25 | [abs](https://arxiv.org/abs/2510.08554) | Low-variance sequence-level RL; outperforms diffu-GRPO on reasoning/planning/coding. |
| **DiFFPO: Reason Fast and Furious via RL** | arXiv'25 | [abs](https://arxiv.org/abs/2510.02212) | Jointly optimizes reasoning quality + few-step sampling. |
| **DMPO: Distribution Matching Policy Optimization** | arXiv'25 | [abs](https://arxiv.org/abs/2510.08233) | Distribution-matching RL objective avoiding mean-field likelihood bias. |
| **TraDo-8B / TraceRL: Trajectory-Aware RL** | arXiv'25 | [abs](https://arxiv.org/abs/2509.06949) · [code](https://github.com/Gen-Verse/dLLM-RL) | Trajectory-aware RL on SDAR block diffusion → TraDo-8B-Thinking, first long-CoT diffusion LLM. |

---

## 8. Multimodal & unified diffusion

### Multimodal diffusion LLMs

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **LLaDA-V: Visual Instruction Tuning** | arXiv'25 | [abs](https://arxiv.org/abs/2505.16933) · [code](https://github.com/ML-GSAI/LLaDA-V) | First purely diffusion-based MLLM (LLaDA + vision encoder); SOTA among diffusion MLLMs. |
| **LaViDa: Large Diffusion LM for Multimodal Understanding** | arXiv'25 | [abs](https://arxiv.org/abs/2505.16839) | Complementary masking + prefix KV-cache for fast controllable multimodal understanding. |
| **Dimple: Discrete Diffusion MLLM w/ Parallel Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2505.16990) | First discrete-diffusion MLLM; AR-then-diffusion recipe + confident parallel decoding. |
| **MMaDA: Multimodal Large Diffusion LMs** | NeurIPS'25 | [abs](https://arxiv.org/abs/2505.15809) · [code](https://github.com/Gen-Verse/MMaDA) · [demo](https://huggingface.co/spaces/Gen-Verse/MMaDA) | Modality-agnostic unified diffusion; mixed-CoT tuning + UniGRPO RL (text + understanding + T2I). |
| **MMaDA-Parallel: Thinking-Aware Editing & Generation** | arXiv'25 | [abs](https://arxiv.org/abs/2511.09611) | Parallel thinking-aware decoding for interleaved reasoning, image editing, generation. |
| **Lavida-O: Elastic Masked Diffusion for Unified Understanding & Generation** | arXiv'25 | [abs](https://arxiv.org/abs/2509.19244) | Elastic mixture-of-transformers masked diffusion; grounding + 1024px generation. |
| **Lumina-DiMOO: Omni Diffusion LLM** | arXiv'25 | [abs](https://arxiv.org/abs/2510.06308) | Fully discrete omni-diffusion across all modalities; SOTA among open unified models. |

### Unified any-to-any diffusion

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **UniDisc: Unified Multimodal Discrete Diffusion** | arXiv'25 | [abs](https://arxiv.org/abs/2503.20853) | Joint text+image masked discrete diffusion: joint generation, inpainting, editing. |
| **Transfusion: Next-Token + Diffuse Images, One Model** | arXiv'24 | [abs](https://arxiv.org/abs/2408.11039) | Single transformer: next-token loss (text) + continuous diffusion (images) at 7B. |
| **Show-o: Unify Understanding & Generation** | arXiv'24 | [abs](https://arxiv.org/abs/2408.12528) | AR text + discrete diffusion images in one transformer. |
| **Show-o2: Native Unified Multimodal Models** | arXiv'25 | [abs](https://arxiv.org/abs/2506.15564) | Scales to image+video via 3D causal VAE + flow-matching heads. |
| **CoDi: Any-to-Any via Composable Diffusion** | NeurIPS'23 | [abs](https://arxiv.org/abs/2305.11846) | Bridges per-modality latents to condition/generate any mix of text/image/video/audio. |
| **JanusFlow: Autoregression + Rectified Flow** | arXiv'24 | [abs](https://arxiv.org/abs/2411.07975) | Rectified flow inside an LLM with decoupled encoders for unified understanding+generation. |
| **FUDOKI: Discrete Flow-Matching Unified MLLM** | NeurIPS'25 | [abs](https://arxiv.org/abs/2505.20147) | Purely discrete-flow-matching unified MLLM with self-correcting understanding + image generation. |
| **Muddit: Unified Discrete Diffusion w/ T2I Priors** | arXiv'25 | [abs](https://arxiv.org/abs/2505.23606) | Reuses pretrained T2I visual priors for fast parallel text+image generation. |

### Vision-Language-Action (robotics)

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Discrete Diffusion VLA** | arXiv'25 | [abs](https://arxiv.org/abs/2508.20072) | First discrete-diffusion action decoding in one VLA transformer; 96.3% LIBERO with fewer evals. |
| **dVLA: Diffusion VLA w/ Multimodal CoT** | arXiv'25 | [abs](https://arxiv.org/abs/2509.25681) | Built on MMaDA + FAST action tokens for interpretable manipulation policies. |
| **Unified Diffusion VLA** | arXiv'25 | [abs](https://arxiv.org/abs/2511.01718) | Joint discrete denoising: understand, predict future images, and generate actions in one process. |
| **Dream-VL & Dream-VLA** | arXiv'25 | [abs](https://arxiv.org/abs/2512.22615) | Diffusion-backbone VLM/VLA; bidirectionality suits action chunking (97.2% LIBERO). |

---

## 9. Inference & acceleration

The KV-cache + parallel-decoding problem is *the* practical bottleneck for dLLMs.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Fast-dLLM: KV Cache + Parallel Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2505.22618) · [code](https://github.com/NVlabs/Fast-dLLM) | Block-wise approximate KV cache + confidence-aware parallel decoding: up to **27.6×** throughput, training-free. |
| **Fast-dLLM v2: Efficient Block-Diffusion LLM** | arXiv'25 | [abs](https://arxiv.org/abs/2509.26328) | AR→block-diffusion with ~1B FT tokens (500× less than Dream), AR quality retained. |
| **dKV-Cache: The Cache for Diffusion LMs** | arXiv'25 | [abs](https://arxiv.org/abs/2505.15781) | Delayed, decode-conditioned KV reuse for bidirectional dLLMs; near-lossless. |
| **dLLM-Cache: Adaptive Caching** | arXiv'25 | [abs](https://arxiv.org/abs/2506.06295) | Long-interval prompt + feature-guided response caching: up to 9.1× FLOPs reduction. |
| **SlowFast Sampling: Three Golden Principles** | arXiv'25 | [abs](https://arxiv.org/abs/2506.10848) | Certainty/convergence/positional principles → up to 15.63× speedup on LLaDA. |
| **ReMDM: Remasking + Inference-Time Scaling** | arXiv'25 | [abs](https://arxiv.org/abs/2503.00307) | Principled remasking lets masked diffusion fix earlier tokens & scale with steps. |
| **SDTT: Self-Distillation Through Time** | ICLR'25 | [abs](https://arxiv.org/abs/2410.21035) | Distills 1024-step sampling into <64 steps (32–64× fewer). |
| **dParallel: Learnable Parallel Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2509.26488) | Certainty-forcing distillation → ~30 steps (8.5–10.5× speedup) on LLaDA-8B. |
| **WINO: Wide-In, Narrow-Out Revokable Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2507.18578) | Training-free draft-and-verify with re-masking makes parallel decoding reversible. |
| **Self Speculative Decoding for dLLMs** | arXiv'25 | [abs](https://arxiv.org/abs/2510.04147) | The dLLM drafts & verifies itself via hierarchical verification trees. |
| **ParallelBench: Trade-offs of Parallel Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2510.04767) | Benchmark isolating when parallel token commitment breaks dependencies. |
| **DiMO / DiDi-Instruct / DLM-One: one/few-step generation** | arXiv'25 | [abs](https://arxiv.org/abs/2509.25035) · [abs](https://arxiv.org/abs/2506.00290) | Distill dLLMs into (near) one-step generators for ultra-fast decoding. |
| **Prophet: dLLMs Know the Answer Before Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2508.19982) | Early-commit decoding exploiting answer convergence (top-2 gap); up to 3.4× fewer steps. |
| **EB-Sampler: Entropy-Bounded Unmasking** | arXiv'25 | [abs](https://arxiv.org/abs/2505.24857) | Drop-in sampler unmasking many tokens/step under an entropy budget; 2–3× speedup, no quality loss. |
| **CreditDecoding: Trace Credits** | arXiv'25 | [abs](https://arxiv.org/abs/2510.06133) | Accumulates historical logits to converge underconfident-but-correct tokens; 5.48× on LLaDA-8B. |
| **LocalLeap: Local Determinism Propagation** | arXiv'25 | [abs](https://arxiv.org/abs/2510.07081) | Training-free adaptive parallel decoding around high-confidence anchors. |
| **Set Block Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2509.04185) | Unifies next-token + masked-token prediction; parallel non-consecutive tokens with exact KV-cache. |
| **Learn2PD: Learnable Parallel Decoding** | arXiv'25 | [abs](https://arxiv.org/abs/2509.25188) | Lightweight filter predicts per-token correctness to adaptively unmask; up to 22.58× speedup. |
| **Dilated Scheduling (Plan for Speed)** | arXiv'25 | [abs](https://arxiv.org/abs/2506.19037) | Unmasks well-separated token groups in parallel; up to 5.8× faster, complements per-token samplers. |
| **Learning Unmasking Policies (RL)** | arXiv'25 | [abs](https://arxiv.org/abs/2512.09106) | RL-trained policies decide which tokens to unmask, beating hand-crafted heuristics. |
| **SchED: Progress-Aware Confidence Schedules** | arXiv'25 | [abs](https://arxiv.org/abs/2512.02892) | Training-free early-exit schedule keyed to logit confidence; 3.8–4.0× faster. |
| **FlashDLM / FreeCache** | arXiv'25 | [abs](https://arxiv.org/abs/2505.21467) | Training-free KV caching + guided diffusion; 12.14× speedup, closing gap to AR. |
| **Sparse-dLLM: Dynamic Cache Eviction** | arXiv'25 | [abs](https://arxiv.org/abs/2508.02558) | Sparse attention + delayed bidirectional cache eviction; up to 10× throughput. |
| **d²Cache: Dual Adaptive Caching** | arXiv'25 | [abs](https://arxiv.org/abs/2509.23094) | Selectively updates token KV states for faster *and* higher-quality inference. |
| **CDLM: Consistent Diffusion Language Models** | arXiv'26 | [abs](https://arxiv.org/abs/2605.00161) | Single-stage teacher-free multi-path consistency training; SOTA few-step sampling. |
| **HERALD: CPU-GPU Cooperative KV Serving** | arXiv'26 | [abs](https://arxiv.org/abs/2606.21633) | Serving system offloading/retrieving KV across CPU-GPU for block-diffusion throughput. |

---

## 10. Guidance, control & constrained decoding

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Simple Guidance Mechanisms for Discrete Diffusion** | ICLR'25 | [abs](https://arxiv.org/abs/2412.10193) · [code](https://github.com/kuleshov-group/discrete-diffusion-guidance) | Derives classifier & classifier-free guidance for discrete diffusion + more-controllable uniform-noise variant. |
| **Improving CFG in Masked Diffusion** | arXiv'25 | [abs](https://arxiv.org/abs/2507.08965) | Early guidance hurts, late-stage guidance helps — theory-backed timing. |
| **SVDD: Soft Value-Based Decoding** | arXiv'24 | [abs](https://arxiv.org/abs/2408.08252) | Derivative-free reward guidance for discrete diffusion, no fine-tuning. |
| **Constrained Decoding of dLLMs with CFGs** | arXiv'25 | [abs](https://arxiv.org/abs/2508.10111) | First grammar-constrained decoding for dLLMs (JSON, C++ infilling) via additive infilling. |
| **Unveiling dLLM Potential in Controllable Generation** | arXiv'25 | [abs](https://arxiv.org/abs/2507.04504) | dLLMs exploit bidirectional context for structured/controllable output + faster inference. |

---

## 11. Applications

### Infilling / editing / style / paraphrase / GEC

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **DiffusER: Edit-based Reconstruction** | arXiv'22 | [abs](https://arxiv.org/abs/2210.16886) | Generation as iterative edit ops (insert/delete/replace); revise & condition on prototypes. |
| **DiffuSum: Diffusion-Enhanced Extractive Summarization** | ACL'23 (Findings) | [abs](https://arxiv.org/abs/2305.01735) | Diffusion-generated sentence representations + matching → SOTA on CNN/DailyMail. |
| **DiffuDetox: Mixed Diffusion for Detoxification** | arXiv'23 | [abs](https://arxiv.org/abs/2306.08505) | First diffusion detoxification; mixes conditional + unconditional training. |
| **Fine-grained Text Style Transfer with Diffusion** | arXiv'23 | [abs](https://arxiv.org/abs/2305.19512) | One diffusion LM handles 13 fine-grained StylePTB transfers without parsers. |
| **LDP: Paraphrase via Controllable Latent Diffusion** | arXiv'24 | [abs](https://arxiv.org/abs/2404.08938) | Latent-space diffusion improves paraphrase quality/diversity/efficiency. |
| **Corrective Diffusion Language Models** | arXiv'25 | [abs](https://arxiv.org/abs/2512.15596) | Post-training so dLLMs down-weight wrong tokens & iteratively refine — diffusion-native correction. |

### Machine translation

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **XDLM: Cross-lingual Diffusion LM for MT** | arXiv'23 | [abs](https://arxiv.org/abs/2307.13560) | Cross-lingual pretraining (TLDM) before fine-tuning; beats diffusion & Transformer MT baselines. |
| *(see also DINOISER, SeqDiffuSeq, Difformer, CDCD in §2)* | | | Continuous text-diffusion backbones commonly evaluated on MT. |

### Long-context

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **LongLLaDA: Long Context for Diffusion LLMs** | AAAI'26 | [abs](https://arxiv.org/abs/2506.14429) | First length-extrapolation for dLLMs (training-free NTK-RoPE); stable-perplexity behavior. |
| **UltraLLaDA: 128K Context for dLLMs** | arXiv'25 | [abs](https://arxiv.org/abs/2510.10481) | Post-training + diffusion-aware RoPE scaling to a stable 128K context. |

### Structured generation & dialogue

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **TabDDPM: Tabular Data with Diffusion** | arXiv'22 | [abs](https://arxiv.org/abs/2209.15421) | Canonical diffusion for mixed-type tabular data (Gaussian + multinomial). |
| **DiffusionDialog: Diverse Dialog via Latent Diffusion** | arXiv'24 | [abs](https://arxiv.org/abs/2404.06760) | Latent-space diffusion boosts response diversity in dialogue. |

### Retrieval-augmented generation

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **SPREAD: RAG for Diffusion LMs** | arXiv'26 | [abs](https://arxiv.org/abs/2601.11342) | First systematic RAG-for-dLLM study; fixes "Response Semantic Drift" via relevance-guided denoising. |
| **ARAM: Adaptive Guidance for RAG Masked Diffusion** | arXiv'26 | [abs](https://arxiv.org/abs/2603.17677) | Calibrates guidance by SNR of retrieved-context shift for knowledge-intensive QA. |

---

## 12. Safety, watermarking & robustness

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **PAD: Jailbreaking Large Language Diffusion Models** | arXiv'25 | [abs](https://arxiv.org/abs/2507.19227) | First systematic dLLM jailbreak; Multi-Point Attention Attack exploits parallel denoising. |
| **The Devil behind the Mask: Emergent Safety Vuln of dLLMs** | arXiv'25 | [abs](https://arxiv.org/abs/2507.11097) | Bidirectional/masked generation itself is a novel attack surface. |
| **DiffuGuard: Intrinsic Safety Lost & Found in dLLMs** | arXiv'25 | [abs](https://arxiv.org/abs/2509.24296) | Where dLLMs lose safety during denoising + a training-free defense. |
| **Watermarking Diffusion Language Models** | arXiv'25 | [abs](https://arxiv.org/abs/2509.24368) | First watermark for arbitrary-order diffusion generation. |
| **DMark: Order-Agnostic Watermarking for dLLMs** | arXiv'25 | [abs](https://arxiv.org/abs/2510.02902) | Watermark surviving non-sequential token generation. |

---

## 13. Efficiency: quantization

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Quantization Meets dLLMs: A Systematic PTQ Study** | arXiv'25 | [abs](https://arxiv.org/abs/2508.14896) | First PTQ study for dLLMs; diagnoses activation outliers across bit-width/method/task. |
| **DLLMQuant** | arXiv'25 | [abs](https://arxiv.org/abs/2508.14090) | Fine-tuning-free PTQ designed around the bidirectional/denoising structure. |
| **Quant-dLLM: Extreme Low-Bit (2-bit) PTQ** | arXiv'25 | [abs](https://arxiv.org/abs/2510.03274) | 2-bit PTQ where naive AR quantization transfer fails → edge deployment. |

---

## 14. Embeddings, energy-based & Bayesian flow networks

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **Bayesian Flow Networks** | arXiv'23 | [abs](https://arxiv.org/abs/2308.07037) | Bayesian inference on distribution parameters (differentiable for discrete); beats discrete diffusion on text8. |
| **Unifying BFNs & Diffusion via SDEs** | arXiv'24 | [abs](https://arxiv.org/abs/2404.15766) | Connects BFNs to diffusion via linear SDEs; 5–20× faster solvers. |
| **EDLM: Energy-Based Diffusion LM** | ICLR'25 | [abs](https://arxiv.org/abs/2410.21357) | Sequence-level residual EBM per step fixes token-independence errors; ~1.3× faster. |
| **DiffEmbed: Diffusion LMs as Text Embedders** | EMNLP'25 | [abs](https://arxiv.org/abs/2505.15045) | Bidirectional attention → big gains on long-doc & reasoning-intensive retrieval. |
| **Residual Energy-Based Models for Text** | arXiv'20 | [abs](https://arxiv.org/abs/2004.11714) | Foundational residual-EBM formulation underpinning energy-corrected diffusion LMs. |

---

## 15. "Language of X": biology & speech

Discrete diffusion over non-text token sequences — the techniques transfer directly.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **DPLM: Diffusion Protein Language Models** | ICML'24 | [abs](https://arxiv.org/abs/2402.18567) | Discrete masked diffusion over protein sequences; strong generation + representation, beats ESM2. |
| **DPLM-2: Multimodal Diffusion Protein LM** | arXiv'24 | [abs](https://arxiv.org/abs/2410.13782) | Jointly diffuses sequence + tokenized 3D structure (co-generation, folding, inverse folding). |
| **EvoDiff: Protein Generation with Evolutionary Diffusion** | bioRxiv'23 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.11.556673) | Evolutionary-scale discrete diffusion protein design in sequence space. |
| **Discrete Diffusion for Text-Aligned Speech Tokens** | arXiv'25 | [abs](https://arxiv.org/abs/2509.20060) | Replaces AR speech-token decoder with discrete diffusion → better reconstruction, ASR, speed. |
| **DiSTAR: Discrete Diffusion over Token AR Representation** | arXiv'25 | [abs](https://arxiv.org/abs/2510.12210) | Discrete diffusion over a scalable token-AR representation for speech generation. |
| **VocalNet-MDM: Streaming Speech LLM via Masked Diffusion** | arXiv'26 | [abs](https://arxiv.org/abs/2602.08607) | Self-distilled masked diffusion + hierarchical block masking; 3.7–10× decode speedup over AR. |
| **LaDA-Band: Language-Diffusion for Music Accompaniment** | arXiv'26 | [abs](https://arxiv.org/abs/2604.11052) | Language-diffusion model for vocal-to-accompaniment music generation. |

---

## 16. Evaluation & benchmarks

- **Perplexity / likelihood.** dLLMs are evaluated by an ELBO/likelihood **upper bound** on NLL — not directly comparable to AR exact likelihood. See MDLM, MD4, Plaid, SEDD.
- **Reasoning / knowledge.** GSM8K, MATH500, MMLU — reported by LLaDA, Dream, d1, DiffuCoder.
- **Coding.** HumanEval(+), MBPP(+), EvalPlus — DiffuCoder, Dream-Coder, Seed Diffusion.
- **Planning.** Countdown, Sudoku — MGDM/"Beyond Autoregression".
- **Speed / throughput.** tok/s on H100 — Mercury, Gemini Diffusion, Seed Diffusion, Fast-dLLM.
- **Parallel-decoding fidelity.** [ParallelBench](https://arxiv.org/abs/2510.04767) isolates dependency breakage under parallel commitment.
- **Honest baselines.** [Promises, Outlooks and Challenges](https://arxiv.org/abs/2406.11473) benchmarks SEDD vs GPT-2 on quality *and* latency.

---

## 17. Extended index — deep gap-sweep (2022–2026)

An additional **125 papers** surfaced by an aggressive multi-agent gap-sweep and individually link-verified. Grouped by topic, oldest→newest. See [PAPERS.md](PAPERS.md) for the full sortable master table of every paper in this repo.

### 17 · 10. Guidance, control & constrained decoding

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2024-06 | **Unlocking Guidance for Discrete State-Space Diffusion and Flow Models** | [abs](https://arxiv.org/abs/2406.01572) | Introduces Discrete Guidance, a principled CTMC-based method that makes classifier and classifier-free guidance tractable for discrete-state diffusion and flow models. |
| 2024-10 | **Plug-and-Play Controllable Generation for Discrete Masked Models** | [abs](https://arxiv.org/abs/2410.02143) | Proposes a training-free, gradient-free importance-sampling framework to steer masked discrete models toward posteriors, constraints, or reward objectives. |
| 2024-10 | **Steering Masked Discrete Diffusion Models via Discrete Denoising Posterior Prediction** | [abs](https://arxiv.org/abs/2410.08134) | DDPP casts steering pretrained masked diffusion models as Bayesian posterior inference with simulation-free objectives that work for non-differentiable rewards. |
| 2024-10 | **Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design** | [abs](https://arxiv.org/abs/2410.13643) | DRAKES enables reward-guided control by backpropagating rewards through Gumbel-Softmax-relaxed discrete diffusion trajectories while KL-regularizing toward the base model. |
| 2025-01 | **A General Framework for Inference-time Scaling and Steering of Diffusion Models** | [abs](https://arxiv.org/abs/2501.06848) | Feynman-Kac steering resamples interacting particles by reward potentials, giving gradient-free attribute and quality control for text (and image) diffusion models. |
| 2025-02 | **Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods** | [abs](https://arxiv.org/abs/2502.01384) | SEPO is a policy-gradient (RLHF-style) algorithm for reward-guided fine-tuning of discrete diffusion LMs that handles non-differentiable rewards. |
| 2025-03 | **Constrained Discrete Diffusion** | [abs](https://arxiv.org/abs/2503.09790) | Embeds differentiable constraint optimization into the diffusion sampler for training-free, zero-violation constrained language and molecule generation. |
| 2025-05 | **Test-Time Alignment of Discrete Diffusion Models with Sequential Monte Carlo** | [abs](https://arxiv.org/abs/2505.22524) | Derives twisted SMC importance weights and locally optimal proposals for training-free reward-aligned inference-time steering of discrete diffusion models. |
| 2025-06 | **What Exactly Does Guidance Do in Masked Discrete Diffusion Models** | [abs](https://arxiv.org/abs/2506.10971) | Provides the theory of classifier-free guidance in masked discrete diffusion, solving the guided reverse dynamics and characterizing its effect on the sampled distribution. |
| 2025-12 | **Activation Steering for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2512.24143) | Introduces an activation-steering primitive that extracts a low-dimensional direction and intervenes on residual activations to control MDLM behavior without optimization. |
| 2026-01 | **ILRR: Inference-Time Steering Method for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2601.21647) | A learning-free framework that steers DLMs by aligning internal activations to a single reference sequence, enabling attribute control on LLaDA and MDLM. |
| 2026-05 | **Primal-Dual Guided Decoding for Constrained Discrete Diffusion** | [abs](https://arxiv.org/abs/2605.09749) | Formulates constrained generation as KL-regularized optimization solved online with adaptive Lagrangian multipliers, adding constraint-dependent logit biases at each denoising step. |
| 2026-05 | **Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.10971) | Uses sparse-autoencoder analysis to show attributes commit on distinct schedules, then adaptively concentrates steering interventions on the forming steps to avoid quality loss. |
| 2026-05 | **Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.13043) | A plug-and-play inference-time defense that detects harmful tokens, remasks them, and resumes denoising with contrastive safety-direction steering scaled by harmfulness. |
| 2026-05 | **Constrained Code Generation with Discrete Diffusion** | [abs](https://arxiv.org/abs/2605.16829) | CDC is a training-free neurosymbolic sampler that injects constraint-aware denoising operators into discrete diffusion to enforce correctness, security, and syntax in code. |
| 2026-05 | **DLM-SWAI: Steering Diffusion Language Models Before They Unmask** | [abs](https://arxiv.org/abs/2605.29626) | A training-free steering method that biases per-step token distributions with pre-computed token-level style scores for controllable DLM generation without auxiliary models. |

### 17 · 11. Applications

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2022-12 | **Diffusion Glancing Transformer for Parallel Sequence to Sequence Learning** | [abs](https://arxiv.org/abs/2212.10240) | Introduces a discrete modality-diffusion process with residual glancing sampling for non-autoregressive machine translation, improving translation accuracy while keeping fast parallel decoding (NAACL 2024). |
| 2023-06 | **PoetryDiffusion: Towards Joint Semantic and Metrical Manipulation in Poetry Generation** | [abs](https://arxiv.org/abs/2306.08456) | First diffusion model for jointly controlling semantics and metrics in poetry generation, explicitly covering non-English Chinese SongCi alongside English sonnets (AAAI 2024). |
| 2024-02 | **Text Diffusion with Reinforced Conditioning** | [abs](https://arxiv.org/abs/2402.14843) | Text diffusion with reinforced self-conditioning and time-aware variance scaling, evaluated on IWSLT14 De-En and WMT14 En-De machine translation (AAAI 2024). |
| 2024-07 | **Discrete Diffusion Language Model for Efficient Text Summarization** | [abs](https://arxiv.org/abs/2407.10998) | Makes discrete diffusion work for long abstractive summarization via a semantic-aware noising process and CrossMamba, with faster-than-autoregressive inference. |
| 2024-09 | **An Effective Deployment of Diffusion LM for Data Augmentation in Low-Resource Sentiment Classification** | [abs](https://arxiv.org/abs/2409.03203) | DiffusionCLS uses a diffusion LM to reconstruct strong label-related tokens and generate pseudo-samples for low-resource, imbalanced sentiment classification (EMNLP 2024). |
| 2024-09 | **Table-to-Text Generation with Pretrained Diffusion Models** | [abs](https://arxiv.org/abs/2409.13739) | Adapts pretrained diffusion language models to data-to-text/table-to-text, achieving a better quality-diversity balance than autoregressive text-to-text baselines. |
| 2024-10 | **Discrete Modeling via Boundary Conditional Diffusion Processes** | [abs](https://arxiv.org/abs/2410.22380) | Adds discrete-boundary guidance to continuous diffusion and sets new state-of-the-art over prior continuous diffusion language models on three machine-translation tasks (NeurIPS 2024). |
| 2024-10 | **Private Synthetic Text Generation with Diffusion Models** | [abs](https://arxiv.org/abs/2410.22971) | First systematic study of whether non-autoregressive text diffusion models are better than AR LLMs at generating synthetic training text under differential privacy. |
| 2024-11 | **DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models** | [abs](https://arxiv.org/abs/2411.03250) | A VAE-plus-latent-diffusion framework that lets a diffusion language model synthesize high-quality structured data (tabular, code, tool-call) whose downstream utility can exceed real data (ACL 2025 Findings). |
| 2025-10 | **DiffGRM: Diffusion-based Generative Recommendation Model** | [abs](https://arxiv.org/abs/2510.21805) | Replaces the autoregressive decoder in generative recommendation with a masked discrete diffusion model for any-order parallel semantic-ID generation, improving NDCG@10 by 6.9-15.5%. |
| 2025-11 | **LLaDA-Rec: Discrete Diffusion for Parallel Semantic ID Generation in Generative Recommendation** | [abs](https://arxiv.org/abs/2511.06254) | Reformulates generative recommendation as parallel semantic-ID generation with discrete diffusion, using bidirectional attention to remove autoregressive error accumulation. |
| 2025-11 | **DiffuGR: Generative Document Retrieval with Diffusion Language Models** | [abs](https://arxiv.org/abs/2511.08150) | First generative document retrieval method to cast DocID generation as a discrete diffusion process, enabling parallel decoding with tunable speed/accuracy tradeoffs. |
| 2025-11 | **Masked Diffusion for Generative Recommendation** | [abs](https://arxiv.org/abs/2511.23021) | Models user semantic-ID sequences with masked diffusion (MaskGR), beating TIGER by ~21.9% NDCG@5 and excelling in data-sparse recommendation settings. |
| 2026-01 | **Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation (Extended Version)** | [abs](https://arxiv.org/abs/2601.07152) | Aligns diffusion language models for schema-adherent structured JSON generation via language-mediated multi-agent RL, achieving the highest task success rate over diffusion and AR baselines. |
| 2026-01 | **Style Transfer as Bias Mitigation: Diffusion Models for Synthetic Mental Health Text for Arabic** | [abs](https://arxiv.org/abs/2601.14124) | Uses pretraining-free diffusion style transfer to generate gender-balanced synthetic Arabic mental-health text, a low-resource non-English text-diffusion application. |
| 2026-01 | **Masked Diffusion Generative Recommendation** | [abs](https://arxiv.org/abs/2601.19501) | Industrial masked-diffusion recommendation framework (MDGR) with a parallel codebook and two-stage parallel decoding, validated by online A/B tests showing +3.69% GMV. |
| 2026-02 | **Sentence Curve Language Models** | [abs](https://arxiv.org/abs/2602.01807) | Extends diffusion language models to predict spline sentence curves instead of static word embeddings, achieving state-of-the-art among diffusion LMs on the IWSLT14 and WMT14 translation benchmarks. |
| 2026-02 | **DLLM Agent: See Farther, Run Faster** | [abs](https://arxiv.org/abs/2602.07451) | Shows diffusion-LLM backbones in a matched agent workflow induce different planning/tool-use behavior and deliver over 30% end-to-end speedups versus autoregressive agents at comparable accuracy. |
| 2026-02 | **DiffuRank: Effective Document Reranking with Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.12528) | First to use diffusion language models for document reranking, leveraging bidirectional attention and parallel decoding to match or beat autoregressive rerankers on TREC DL and BEIR. |
| 2026-03 | **Diffutron: A Masked Diffusion Language Model for Turkish Language** | [abs](https://arxiv.org/abs/2603.20466) | A Turkish-specific masked diffusion language model built via LoRA continual pre-training and progressive instruction tuning of a multilingual encoder for non-autoregressive Turkish generation. |
| 2026-05 | **DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.07210) | Uses a diffusion LM's native masked-position prediction to produce multiple representative retrieval tokens in a single bidirectional pass, topping BEIR-7 dense-retrieval baselines. |
| 2026-05 | **Are Diffusion Language Models Good Database Analysts?** | [abs](https://arxiv.org/abs/2605.27791) | First systematic study of diffusion language models for text-to-SQL, introducing the SQL-D1 agentic framework and showing structural-robustness advantages over left-to-right decoding. |
| 2026-06 | **Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2606.04535) | Training-free method that dynamically sizes end-anchors so diffusion LLMs produce parseable JSON/reasoning templates without truncation, boosting GSM8K and MATH accuracy. |

### 17 · 12. Safety, watermarking & robustness

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-08 | **Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position** | [abs](https://arxiv.org/abs/2508.12398) | First safety-alignment analysis of dLLMs, showing middle tokens (not first tokens) drive safety and proposing MOSA middle-token alignment. |
| 2026-01 | **LR-DWM: Efficient Watermarking for Diffusion Language Models** | [abs](https://arxiv.org/abs/2601.12376) | Low-overhead watermark that biases tokens on both left and right neighbors, tailored to the bidirectional decoding of dLLMs. |
| 2026-01 | **Membership Inference Attacks Against Fine-tuned Diffusion Language Models** | [abs](https://arxiv.org/abs/2601.20125) | First systematic MIA study on dLLMs, with SAMA exploiting multiple maskable configurations for large AUC gains at low false-positive rates. |
| 2026-02 | **Safer by Diffusion, Broken by Context: Diffusion LLM's Safety Blessing and Its Failure Mode** | [abs](https://arxiv.org/abs/2602.00388) | Shows dLLMs have intrinsic jailbreak robustness but a context-nesting attack bypasses it, including the first known jailbreak of Gemini Diffusion. |
| 2026-02 | **Step-Wise Refusal Dynamics in Autoregressive and Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.02600) | Comparative study finding diffusion sampling improves jailbreak robustness over AR, plus an internal-signal detector for jailbreak attempts. |
| 2026-02 | **TDGNet: Hallucination Detection in Diffusion Language Models via Temporal Dynamic Graphs** | [abs](https://arxiv.org/abs/2602.08048) | Formulates dLLM hallucination detection as learning over evolving token-level attention graphs to capture how evidence emerges during denoising. |
| 2026-03 | **DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning** | [abs](https://arxiv.org/abs/2603.16459) | Uses token-level uncertainty evolution and trajectory-deviation analysis over the denoising process to flag factual errors in dLLM outputs. |
| 2026-04 | **Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.08557) | TrajHijack re-masks committed refusal tokens for 74-82% ASR and reveals a Defense Inversion Effect where safety training worsens vulnerability. |
| 2026-04 | **HIVE: Hidden-Evidence Verification for Hallucination Detection in Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2604.26139) | Detects hallucinations by extracting compressed hidden evidence from denoising trajectories, reaching up to 0.92 AUROC across QA benchmarks. |
| 2026-05 | **The Safety-Aware Denoiser for Text Diffusion Models** | [abs](https://arxiv.org/abs/2605.08116) | Inference-time defense that steers the denoising trajectory toward provably safe text regions, covering hazards, memorization, and jailbreak resistance. |
| 2026-05 | **Backdooring Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.19262) | First training-time backdoor for MDLMs (SHADOWMASK), replacing the all-mask terminal prior with a trigger-mask mixture to hide a denoising pathway. |
| 2026-06 | **MaskForge: Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2606.04027) | Fully black-box adaptive red-teaming that optimizes over a growing library of structural mask patterns specific to dLLM denoising. |
| 2026-06 | **Global Sketch-Based Watermarking for Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.04486) | Watermarks masked dLLMs via a global vector-valued sketch over the whole sequence, using joint sampling of unresolved positions unavailable to AR schemes. |

### 17 · 15. "Language of X": biology & speech

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2024-02 | **DiscDiff: Latent Diffusion Model for DNA Sequence Generation** | [abs](https://arxiv.org/abs/2402.06079) | Encodes discrete DNA into a continuous latent space so diffusion can generate genomic sequences, and releases the multi-species EPD-GenDNA benchmark of 160k sequences. |
| 2024-02 | **Text-Guided Molecule Generation with Diffusion Language Model** | [abs](https://arxiv.org/abs/2402.13040) | Pioneering diffusion language model (TGM-DLM) that iteratively denoises SMILES token embeddings for text-conditioned molecule generation, outperforming autoregressive MolT5-Base (AAAI'24). |
| 2024-03 | **Diffusion on language model encodings for protein sequence generation** | [abs](https://arxiv.org/abs/2403.03726) | DiMA runs continuous diffusion over protein-LM (ESM) encodings, producing novel, diverse, high-quality sequences that beat autoregressive, discrete-diffusion, and flow-matching baselines. |
| 2024-04 | **Guided Discrete Diffusion for Electronic Health Record Generation** | [abs](https://arxiv.org/abs/2404.12314) | Applies discrete diffusion with guidance to the discrete token language of tabular medical codes for privacy-preserving synthetic EHR generation (TMLR'25). |
| 2024-09 | **Latent Diffusion Models for Controllable RNA Sequence Generation** | [abs](https://arxiv.org/abs/2409.09828) | RNAdiffusion applies latent diffusion over BERT-encoded RNA representations with reward-guided sampling to generate and optimize variable-length RNA sequences by function. |
| 2024-10 | **MeMDLM: De Novo Membrane Protein Design with Masked Discrete Diffusion Protein Language Models** | [abs](https://arxiv.org/abs/2410.16735) | Adapts the MDLM masked-diffusion framework to protein sequences for de novo membrane-protein design, a domain where structure prediction is scarce. |
| 2024-12 | **PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion** | [abs](https://arxiv.org/abs/2412.17780) | Masked discrete diffusion (MDLM) over peptide SMILES with a bond-dependent masking schedule and Monte Carlo Tree Guidance to co-optimize binding, permeability, and solubility. |
| 2025-05 | **DiffER: Categorical Diffusion for Chemical Retrosynthesis** | [abs](https://arxiv.org/abs/2505.23721) | Template-free categorical (discrete) diffusion that predicts the entire product/reactant SMILES sequence in unison for reaction/retrosynthesis, reaching state-of-the-art top-1 accuracy among template-free methods. |
| 2025-06 | **Diffusion Sequence Models for Enhanced Protein Representation and Generation** | [abs](https://arxiv.org/abs/2506.08293) | DSM is a masked-diffusion protein language model delivering both strong representation learning and generative protein design in one sequence model. |
| 2025-08 | **Cross-Modality Controlled Molecule Generation with Diffusion Language Model** | [abs](https://arxiv.org/abs/2508.14748) | SMILES-based diffusion language model (CMCM-DLM) that composes structure and property constraints via classifier-free and classifier-based guidance without retraining per constraint. |
| 2026-03 | **D3LM: A Discrete DNA Diffusion Language Model for Bidirectional DNA Understanding and Generation** | [abs](https://arxiv.org/abs/2603.01780) | Masked discrete diffusion unifies bidirectional DNA representation learning and generation, outperforming left-to-right autoregressive DNA models on regulatory-element generation. |
| 2026-04 | **CAGenMol: Condition-Aware Diffusion Language Model for Goal-Directed Molecular Generation** | [abs](https://arxiv.org/abs/2604.11483) | Condition-aware discrete diffusion over molecular sequences coupled with reinforcement learning to align generation with non-differentiable drug-design objectives while preserving validity. |
| 2026-06 | **Diffusion-Proof: Recipe for Formal Theorem Proving Beyond Auto-Regressive Generation** | [abs](https://arxiv.org/abs/2606.19315) | First framework to train diffusion LLMs for formal theorem-proving sequences, using bidirectional in-filling for whole-proof writing and local proof correction. |

### 17 · 16. Evaluation & benchmarks

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-02 | **Theoretical Benefit and Limitation of Diffusion Language Model** | [abs](https://arxiv.org/abs/2502.09622) | Analytically shows masked diffusion optimizes perplexity efficiently but needs sampling steps scaling linearly with sequence length to guarantee sequence-level correctness. |
| 2025-08 | **Empirical Analysis of Decoding Biases in Masked Diffusion Models** | [abs](https://arxiv.org/abs/2508.13021) | ACL'26 study identifying rigid-boundary and trivial-token decoding biases in uncertainty-based samplers and linking an 'attention floating' pattern to the strong in-context-learning behavior of masked diffusion models. |
| 2025-10 | **Exploring the Power of Diffusion Large Language Models for Software Engineering: An Empirical Investigation** | [abs](https://arxiv.org/abs/2510.04605) | Empirically benchmarks dLLMs across the software-engineering lifecycle on 52,937 tasks (code generation, defect detection, program repair), reporting large accuracy gains over AR-LLMs. |
| 2025-10 | **Beyond Surface Reasoning: Unveiling the True Long Chain-of-Thought Capacity of Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2510.09544) | Evaluates long chain-of-thought reasoning in dLLMs and identifies a Parallel-Sequential Contradiction where models only exhibit genuine parallelism on easy outputs and revert to AR-like behavior on hard reasoning. |
| 2025-10 | **Attention Sinks in Diffusion Language Models** | [abs](https://arxiv.org/abs/2510.15731) | Analyzes attention-sink behavior specific to masked diffusion LMs, showing sink positions are dynamic yet generation is more robust to sink removal than in AR models. |
| 2025-10 | **How Efficient Are Diffusion Language Models? A Critical Examination of Efficiency Evaluation Practices** | [abs](https://arxiv.org/abs/2510.18480) | Audits throughput/latency benchmarking of diffusion LMs, exposes inconsistent efficiency reporting, and shows under controlled conditions that AR models generally still out-throughput dLLMs. |
| 2026-01 | **Revealing the Attention Floating Mechanism in Masked Diffusion Models** | [abs](https://arxiv.org/abs/2601.07894) | Mechanistically characterizes how masked-diffusion attention forms dynamic, dispersed anchors that shift across denoising steps (vs. fixed AR sinks), explaining their knowledge-task advantage. |
| 2026-01 | **Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2601.14758) | Uses circuit-level mechanistic interpretability to show AR-to-diffusion conversion reorganizes internal computational mechanisms rather than merely swapping the generation procedure. |
| 2026-01 | **Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow** | [abs](https://arxiv.org/abs/2601.15593) | Large-scale evaluation of eight MDLMs up to 100B parameters across 58 benchmarks, quantifying how far current models fall short of their theoretical parallelism and flexible-order advantages versus autoregressive baselines. |
| 2026-01 | **Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from $k$-Parity** | [abs](https://arxiv.org/abs/2601.22450) | Studies how the masked-diffusion objective reshapes learning dynamics, using k-parity as a controlled probe to explain and improve generalization without grokking. |
| 2026-02 | **DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders** | [abs](https://arxiv.org/abs/2602.05859) | First sparse-autoencoder interpretability framework for DLMs (Dream-7B, LLaDA-8B), extracting interpretable features that enable diffusion-time interventions outperforming standard LLM steering. |
| 2026-04 | **Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.00375) | Diagnoses why confidence-based filtering raises single-sample quality but collapses multi-sample diversity, exposing a fundamental quality-exploration tradeoff in DLM decoding. |
| 2026-04 | **Generative Frontiers: Why Evaluation Matters for Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.02718) | Argues generative perplexity alone gives unreliable dLLM rankings and proposes 'generative frontiers,' decomposing perplexity and entropy as the two KL-divergence components into a principled evaluation framework. |
| 2026-04 | **Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2604.10556) | First controlled comparative study of hallucination in dLLMs versus AR models, evaluating knowledge recall, long-form factual consistency, and knowledge-boundary detection and cataloguing dLLM-specific failure modes. |
| 2026-04 | **Measuring Temporal Linguistic Emergence in Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.23235) | Probes when linguistic information becomes decodable across denoising steps, finding coarse semantic features stabilize earlier than precise lexical identity. |
| 2026-05 | **Understanding and Accelerating the Training of Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.13026) | Identifies locality bias in language as the root cause of slow MDM training and derives a bell-shaped time-sampling schedule for ~4x faster convergence. |
| 2026-05 | **Extracting Training Data from Diffusion Language Models via Infilling** | [abs](https://arxiv.org/abs/2605.24173) | Probes memorization in dLLMs, introducing infilling-based extraction that shows edge-conditioned masks extract up to three times more verbatim training data than prefix-only probing, revealing evaluation gaps unique to bidirectional dLLMs. |
| 2026-05 | **TUBE: Tangent Upper Bound on Evidence for Discrete Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.24292) | Provides a variational upper bound on log-likelihood with an unbiased Monte Carlo estimator, complementing ELBOs so likelihood/perplexity of discrete diffusion LMs can be bracketed and fairly compared to autoregressive baselines. |
| 2026-06 | **Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics** | [abs](https://arxiv.org/abs/2606.08417) | Demonstrates that naive samplers can achieve state-of-the-art generative perplexity while producing incoherent text, showing the metric is unsound for diffusion/flow LMs and advocating distributional-divergence evaluation suites. |
| 2026-06 | **Re-evaluating Confidence Remasking in Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.12232) | Rigorously re-examines confidence-based remasking and shows it brings little-to-no benefit over confidence-based unmasking while worsening diversity collapse, highlighting the need for careful decoding-configuration-aware evaluation. |
| 2026-06 | **Diffusion Language Models: An Experimental Analysis** | [abs](https://arxiv.org/abs/2606.19475) | Systematically evaluates eight state-of-the-art dLLMs across eight benchmarks (reasoning, coding, translation, knowledge, problem solving) while dissecting the effect of denoising steps, context length, block size, and unmasking on the quality-efficiency trade-off. |

### 17 · 2. Early & continuous text diffusion

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2022-06 | **Latent Diffusion Energy-Based Model for Interpretable Text Modeling** | [abs](https://arxiv.org/abs/2206.05895) | ICML'22 work fusing a latent-space energy-based model with a diffusion sampler for interpretable, controllable text generation, an early and rarely-listed latent text-diffusion variant. |
| 2022-08 | **Composable Text Controls in Latent Space with ODEs** | [abs](https://arxiv.org/abs/2208.00638) | EMNLP'23 LatentOps framework using ODE-based sampling in a compact text latent space for plug-and-play composable attribute control. |
| 2022-10 | **Categorical SDEs with Simplex Diffusion** | [abs](https://arxiv.org/abs/2210.14784) | Early DeepMind note diffusing directly on the probability simplex via a Cox-Ingersoll-Ross process, an alternative continuous route for categorical/text data. |
| 2022-11 | **Score-based Continuous-time Discrete Diffusion Models** | [abs](https://arxiv.org/abs/2211.16750) | ICLR'23 foundational discrete-diffusion method extending score matching to categorical data via a continuous-time Markov jump process, predating SEDD. |
| 2023-04 | **GlyphDiffusion: Text Generation as Image Generation** | [abs](https://arxiv.org/abs/2304.12519) | Obscure approach that renders target text as glyph images so continuous image diffusion can be applied to discrete text generation. |
| 2023-05 | **DiffuSIA: A Spiral Interaction Architecture for Encoder-Decoder Text Diffusion** | [abs](https://arxiv.org/abs/2305.11517) | Lesser-cited spiral encoder-decoder interaction architecture for conditional text diffusion, evaluated on paraphrase, simplification, question generation, and dialogue. |
| 2023-06 | **PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model** | [abs](https://arxiv.org/abs/2306.02531) | NeurIPS'23 model that couples latent semantic diffusion with an autoregressive decoder to generate paragraphs coarse-to-fine, giving global control while staying fluent. |
| 2023-09 | **Diffusion on the Probability Simplex** | [abs](https://arxiv.org/abs/2309.02530) | Proposes softmax-of-Ornstein-Uhlenbeck diffusion on the simplex, giving a categorical-distribution interpretation for discrete/text data. |
| 2023-10 | **InfoDiffusion: Information Entropy Aware Diffusion Process for Non-Autoregressive Text Generation** | [abs](https://arxiv.org/abs/2310.11976) | EMNLP'23 Findings paper introducing an information-entropy-aware noise schedule with a keyinfo-first generation order for non-autoregressive text diffusion. |
| 2024-02 | **TEncDM: Understanding the Properties of the Diffusion Model in the Space of Language Model Encodings** | [abs](https://arxiv.org/abs/2402.19097) | Runs continuous diffusion over contextual language-model encodings instead of raw embeddings, outperforming prior non-autoregressive text-diffusion baselines (AAAI'25). |
| 2024-03 | **Language Rectified Flow: Advancing Diffusion Language Generation with Probabilistic Flows** | [abs](https://arxiv.org/abs/2403.16995) | NAACL'24 work applying rectified-flow probabilistic flows to language generation as a faster alternative to standard continuous text-diffusion sampling. |
| 2024-08 | **Diffusion Guided Language Modeling** | [abs](https://arxiv.org/abs/2408.04220) | ACL'24 Findings paper using a guided diffusion latent proposal to steer an autoregressive LM's attributes, combining AR fluency with diffusion plug-and-play control. |

### 17 · 4. Scaling laws & training

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-05 | **Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models** | [abs](https://arxiv.org/abs/2505.21400) | NeurIPS'25 information-theoretic convergence theory proving diffusion LM KL sampling error decays with iterations and scales with inter-token mutual information, giving provable speedups over AR's sequential bottleneck even when iterations are fewer than sequence length. |
| 2025-10 | **What Makes Diffusion Language Models Super Data Learners?** | [abs](https://arxiv.org/abs/2510.04071) | Empirically isolates random token masking as the mechanism behind diffusion's superior data efficiency over AR under repeated-data training, dissecting the 'super data learner' crossover. |
| 2025-11 | **Masked Diffusion Models are Secretly Learned-Order Autoregressive Models** | [abs](https://arxiv.org/abs/2511.19152) | Shows the masked-diffusion training objective decomposes into weighted autoregressive losses over token orderings, letting MDMs learn favorable decoding orders and sharpening the diffusion-AR equivalence theory. |
| 2026-01 | **Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation** | [abs](https://arxiv.org/abs/2601.13228) | Reformulates diffusion-style training as any-order/any-subset AR (A3), arguing AR's probabilistic rigor matches diffusion's parallel/bidirectional flexibility and quantifying the expressivity-flexibility trade-off between the paradigms. |
| 2026-01 | **Auto-Regressive Masked Diffusion Models** | [abs](https://arxiv.org/abs/2601.16971) | Directly targets the masked-diffusion-vs-AR performance gap, unifying AR training efficiency with diffusion parallel generation to reach comparable quality with fewer training steps. |
| 2026-03 | **Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison** | [abs](https://arxiv.org/abs/2603.22075) | Matched-data/compute/hardware head-to-head (AR vs MDLM) showing near-identical training throughput but different compute-optimal regimes (AR overfits early, diffusion keeps improving) and a diversity-fluency trade-off. |
| 2026-04 | **On the Trainability of Masked Diffusion Language Models via Blockwise Locality** | [abs](https://arxiv.org/abs/2604.24832) | Analyzes why random-masking masked diffusion is harder to train than AR and introduces locality-aware blockwise variants that inject left-to-right bias for more stable optimization. |
| 2026-06 | **Where to Place the Query? Unveiling and Mitigating Positional Bias in In-Context Learning for Diffusion LLMs via Decoding Dynamics** | [abs](https://arxiv.org/abs/2606.19349) | First study of in-context learning positional bias unique to diffusion LLMs, showing query placement strongly affects accuracy and proposing adaptive placement that approaches oracle ICL performance. |

### 17 · 7. RL, post-training & alignment

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-09 | **GIFT: Guided Importance-Aware Fine-Tuning for Diffusion Language Models** | [abs](https://arxiv.org/abs/2509.20863) | Entropy-based importance weighting for SFT of dLLMs (LoRA or full), beating standard SFT on Sudoku/Countdown/GSM8K/MATH-500 across base and instruct models. |
| 2025-10 | **Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs** | [abs](https://arxiv.org/abs/2510.03680) | Diagnoses the EOS-token early-termination failure of instruction-tuned dLLMs and fixes it with a cyclic distinct-padding scheme recoverable via light fine-tuning. |
| 2025-10 | **Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs** | [abs](https://arxiv.org/abs/2510.09885) | Shows a masked-reconstruction fine-tuning objective transfers dLLM knowledge-injection data efficiency (no paraphrase augmentation) to autoregressive LLMs. |
| 2025-10 | **Aligning Diffusion Language Models via Unpaired Preference Optimization** | [abs](https://arxiv.org/abs/2510.23658) | Introduces ELBO-KTO, combining an ELBO log-likelihood surrogate with prospect-theoretic unpaired preference optimization to align LLaDA-8B-Instruct without pairwise data. |
| 2026-03 | **Mask Is What DLLM Needs: A Masked Data Training Paradigm for Diffusion LLMs** | [abs](https://arxiv.org/abs/2603.15803) | Replaces uniform-random masking in dLLM training with an information-density-driven noise scheduler and complementary priority masking, adding ~4% on code/math reasoning. |
| 2026-04 | **TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.00666) | Trajectory-aware SFT that uses an autoregressive teacher to rank token difficulty, improving the accuracy-parallelism trade-off on LLaDA and Dream at low training cost. |
| 2026-05 | **Learnability-Informed Fine-Tuning of Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.22939) | Shows vanilla SFT hurts dLLMs by ignoring token learnability and proposes LIFT, an SFT post-training method giving up to 3x relative gains on AIME'24/'25. |
| 2026-05 | **NaRA: Noise-Aware LoRA for Parameter-Efficient Fine-Tuning of Diffusion LLMs** | [abs](https://arxiv.org/abs/2605.29716) | First noise-conditioned LoRA for dLLMs, using a hypernetwork to vary low-rank updates along the denoising trajectory and beating noise-agnostic PEFT on reasoning and code. |
| 2026-06 | **Knowledge Editing in Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.03924) | Transfers locate-then-edit knowledge editing to masked dLLMs (LLaDA, Dream), diagnosing multi-token edit failure and adding a correction for partially-unmasked states. |
| 2026-06 | **TimeROME-DLM: Temporal Causal Tracing and Low-Rank Inference-Time Knowledge Editing for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.12841) | First training-free, gradient-free inference-time knowledge-editing/unlearning framework for masked dLLMs via temporal causal tracing plus a closed-form low-rank edit memory. |

### 17 · 9. Inference & acceleration

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-10 | **dInfer: An Efficient Inference Framework for Diffusion Language Models** | [abs](https://arxiv.org/abs/2510.08666) | First modular dLLM inference engine (model, diffusion-iteration, decoding, and KV-cache managers) with tensor/expert parallelism, CUDA graphs, and loop-unrolling, hitting >1100 tok/s and 10x over Fast-dLLM. |
| 2025-12 | **Taming the Memory Footprint Crisis: System Design for Production Diffusion LLM Serving** | [abs](https://arxiv.org/abs/2512.17077) | Production dLLM serving system (dLLM-Serve) attacking the logit/activation memory-footprint crisis via logit-aware activation budgeting, a phase-multiplexed scheduler, and head-centric sparse attention for higher throughput and lower tail latency. |
| 2026-01 | **NPU Design for Diffusion Language Model Inference** | [abs](https://arxiv.org/abs/2601.20706) | First ASIC/NPU accelerator built for dLLMs, delivering a dLLM-oriented ISA and compiler, block-adaptive online KV-cache quantization, and a 7nm RTL implementation for non-GEMM-centric denoising. |
| 2026-02 | **Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs** | [abs](https://arxiv.org/abs/2602.00879) | Tackles the memory-bound 'expert explosion' of MoE dLLM parallel decoding with sequence-level coreset expert selection that maximizes expert reuse across a whole decoding block. |
| 2026-03 | **DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention** | [abs](https://arxiv.org/abs/2603.08026) | Training-free inference framework that recomputes attention/FFN only for salient tokens and reuses cached activations for the rest, reaching up to 9.6x throughput on LLaDA and Dream. |
| 2026-03 | **ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping** | [abs](https://arxiv.org/abs/2603.10088) | Training-free acceleration (ICLR'26) that skips low-importance tokens in early layers based on tensor-variation and confidence, raising LLaDA/Dream throughput to 226-308 tokens per second. |
| 2026-05 | **TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload** | [abs](https://arxiv.org/abs/2605.20179) | Exploits temporal stability of expert activations for interval-based, I/O-aware expert offloading, giving lossless training-free throughput gains for MoE dLLMs on GPU-CPU systems. |
| 2026-05 | **BlockBatch: Multi-Scale Consensus Decoding for Efficient Diffusion Language Model Inference** | [abs](https://arxiv.org/abs/2605.29233) | Training-free system that runs multiple block-size branches for one request inside a single batched forward pass, resolving the block-granularity trade-off with 1.33x average speedup over Fast-dLLM. |
| 2026-06 | **Efficient On-Device Diffusion LLM Inference with Mobile NPU** | [abs](https://arxiv.org/abs/2606.13740) | First NPU-aware on-device dLLM framework (LLADA.CPP) using multi-block speculative decoding, dual-path revision, and swap-optimized memory to cut LLaDA-8B smartphone latency 17-42x. |

---

## Resources

### Code & checkpoints

| Project | Code | Checkpoints |
|---|---|---|
| LLaDA | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | [Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) · [Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) · [1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5) |
| LLaDA-V | [ML-GSAI/LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) | [GSAI-ML/LLaDA-V](https://huggingface.co/GSAI-ML/LLaDA-V) |
| Dream 7B | [DreamLM/Dream](https://github.com/DreamLM/Dream) | [Instruct](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) · [Base](https://huggingface.co/Dream-org/Dream-v0-Base-7B) |
| DiffuLLaMA / DiffuGPT | [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) | [diffugpt-s](https://huggingface.co/diffusionfamily/diffugpt-s) |
| SEDD | [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | [sedd-medium](https://huggingface.co/louaaron/sedd-medium) |
| MDLM | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | [mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt) |
| MD4 | [google-deepmind/md4](https://github.com/google-deepmind/md4) | — |
| BD3-LM | [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | [bd3lm-owt](https://huggingface.co/kuleshov-group/bd3lm-owt-block_size4) |
| Diffusion-LM | [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) | — |
| DiffuSeq | [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) | — |
| Plaid | [igul222/plaid](https://github.com/igul222/plaid) | — |
| RADD | [ML-GSAI/RADD](https://github.com/ML-GSAI/RADD) | — |
| Fast-dLLM | [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | — |
| DiffuCoder | [apple/ml-diffucoder](https://github.com/apple/ml-diffucoder) | [DiffuCoder-7B-Instruct](https://huggingface.co/apple/DiffuCoder-7B-Instruct) |
| MMaDA | [Gen-Verse/MMaDA](https://github.com/Gen-Verse/MMaDA) | [MMaDA-8B-Base](https://huggingface.co/Gen-Verse/MMaDA-8B-Base) |

### Libraries & frameworks

- **[dllm](https://github.com/ZHZisZZ/dllm)** — unified training/inference/eval toolkit (Trainer, LoRA, DeepSpeed, FSDP; recipes for LLaDA, Dream, MDLM, BD3-LM).
- **[dLLM-RL / TraceRL](https://github.com/Gen-Verse/dLLM-RL)** — post-training / RL framework for diffusion LLMs.
- **[Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM)** — fully-open diffusion LM stack (pretraining, eval, inference, checkpoints).
- **[discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance)** — classifier(-free) guidance for discrete diffusion.
- **[e2d2](https://github.com/kuleshov-group/e2d2)** — encoder-decoder diffusion LMs (NeurIPS'25).
- **[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)** — foundational score-based SDE reference impl.
- **Related awesome-lists:** [Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) · [awesome-discrete-diffusion-models](https://github.com/kuleshov-group/awesome-discrete-diffusion-models) · [RUCAIBox/Awesome-Text-Diffusion-Models](https://github.com/RUCAIBox/Awesome-Text-Diffusion-Models)

### Datasets & benchmarks

- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — standard pretraining / perplexity corpus.
- [LM1B / One Billion Word](https://huggingface.co/datasets/billion-word-benchmark/lm1b) — sentence-level NLL benchmark.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) ([source](https://github.com/openai/grade-school-math)) — grade-school math reasoning.
- [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) — Python coding benchmark.

### Blogs & talks

- [Diffusion language models — Sander Dieleman](https://sander.ai/2023/01/09/diffusion-language.html) — best conceptual intro.
- [Language Modeling by Estimating Ratios — Aaron Lou (SEDD)](https://aaronlou.com/blog/2024/discrete-diffusion/)
- [Score-based generative modeling — Yang Song](https://yang-song.net/blog/2021/score/)
- [Dream 7B](https://hkunlp.github.io/blog/2025/dream/) · [Dream-Coder](https://hkunlp.github.io/blog/2025/dream-coder/) · [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/) — HKU NLP.
- [Introducing Mercury — Inception Labs](https://www.inceptionlabs.ai/blog/introducing-mercury)
- [Gemini Diffusion — Google](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/)

### Demos

- [Mercury playground](https://chat.inceptionlabs.ai) · [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [LLaDA Space](https://huggingface.co/spaces/multimodalart/LLaDA) · [LLaDA demo page](https://ml-gsai.github.io/LLaDA-demo/) · [MMaDA Space](https://huggingface.co/spaces/Gen-Verse/MMaDA)

---

## Recent trends (2024–2026)

1. **From theory to scale.** SEDD/MDLM/MD4 (2024) closed the perplexity gap; LLaDA/Dream (2025) proved 7–8B masked diffusion rivals AR; LLaDA-MoE and **LLaDA 2.0 (100B)** pushed to MoE and 100B.
2. **Frontier adoption.** Mercury, **Gemini Diffusion**, and Seed Diffusion made speed (1000–2000+ tok/s) the headline selling point.
3. **The hybrid convergence.** Block diffusion (BD3-LM), Eso-LMs, SDAR, and **TiDAR** interpolate AR↔diffusion to recover KV-cache and quality — the lines are blurring.
4. **Inference is the battleground.** KV-cache for bidirectional models (Fast-dLLM, dKV-Cache), parallel-decoding fidelity (ParallelBench, WINO), and few/one-step distillation (SDTT, Duo, DiMO).
5. **RL grows up.** diffu-GRPO (d1) → VRPO, wd1, GDPO, DMPO — stable RL/preference optimization for a non-AR likelihood.
6. **Data efficiency is the strongest theoretical case.** "Diffusion beats AR in data-constrained settings" + "Super Data Learners" argue any-order training is implicit augmentation.
7. **Beyond text.** Multimodal (LLaDA-V, MMaDA, Lumina-DiMOO), proteins (DPLM), speech, and agents (with sober reality checks).
8. **Open problems.** Exact likelihood, robust agentic/tool use, long-context, safety of parallel denoising, and whether *masked* diffusion is even the right objective ("Scaling Beyond Masked Diffusion").

---

## Contributing

PRs welcome! Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** — real links only, verify arXiv IDs, one factual line per entry, no duplicates.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](LICENSE)

To the extent possible under law, contributors have waived all copyright and related or neighboring rights to this work ([CC0 1.0](LICENSE)).
