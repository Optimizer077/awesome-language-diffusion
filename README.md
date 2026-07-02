# Awesome Language Diffusion [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)
![Papers](https://img.shields.io/badge/papers-190%2B-blue)
![Topics](https://img.shields.io/badge/topics-16-orange)
![Last updated](https://img.shields.io/badge/updated-2026--07-informational)

> A curated, aggressively-sourced list of papers, models, code, and resources on **diffusion models for language** — text, code, and discrete-token sequences.
>
> **190+ verified references** spanning foundations → discrete/masked theory → 7B–100B diffusion LLMs → multimodal/VLA, reasoning, coding, RL, inference, safety, and evaluation.

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
