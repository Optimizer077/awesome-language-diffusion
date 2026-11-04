# Awesome Language Diffusion [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)
![Papers](https://img.shields.io/badge/papers-624-blue)
![Topics](https://img.shields.io/badge/topics-19-orange)
![Last updated](https://img.shields.io/badge/updated-2026--10-informational)

> A curated, aggressively-sourced list of papers, models, code, and resources on **diffusion models for language** — text, code, and discrete-token sequences.
>
> **624 verified references** spanning foundations → discrete/masked theory → 7B–100B diffusion LLMs → multimodal/VLA, reasoning, coding, RL, masking & unmasking strategies, inference, safety, and evaluation.

> 📊 **[PAPERS.md](PAPERS.md) — full sortable master index of all 624 papers with years**, newest-first. Read the topical sections below to learn; scan `PAPERS.md` to browse everything by date.

Language diffusion (a.k.a. **diffusion language models, dLLMs, discrete/masked diffusion LMs**) generates text by *iteratively denoising* a whole sequence in parallel instead of predicting one token left-to-right. In 2024–2026 this went from a research curiosity to **8B–100B open models (LLaDA, Dream, LLaDA-MoE/2.x, DiffusionGemma, Nemotron-Labs-Diffusion)** and **frontier products (Mercury 2/3, Gemini Diffusion, Seed Diffusion)** that reach **1000–2000+ tokens/s**.

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
- [17. Masking, noise schedules & unmasking order](#17-masking-noise-schedules--unmasking-order)
- [18. Extended index — deep gap-sweep (2022–2026)](#18-extended-index--deep-gap-sweep-20222026)
- [19. Oct 2026 update — Jul–Oct 2026 refresh](#19-oct-2026-update--juloct-2026-refresh)
- [📊 PAPERS.md — full sortable master index (624 papers, with years)](PAPERS.md)
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
| **LLaDA 2.1 / 2.2** | 100B MoE | M+B | ✅ weights | Token editing (2.1); insert/delete edits for agents (2.2) | [abs](https://arxiv.org/abs/2602.08676) · [repo](https://github.com/inclusionAI/LLaDA2.X) |
| **DiffusionGemma** | 25.2B (3.8B active) MoE | B | ✅ weights | First open frontier-lab dLLM; served in vLLM/SGLang | [abs](https://arxiv.org/abs/2608.00146) · [HF](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |
| **Nemotron-Labs-Diffusion** | 3B / 8B / 14B | B | ✅ weights | AR + diffusion + self-speculation in one model | [abs](https://arxiv.org/abs/2607.05722) · [HF](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion) |
| **iLLaDA** | 8B | M | ✅ weights | Improved LLaDA, 12T tokens | [abs](https://arxiv.org/abs/2606.25331) · [HF](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct) |
| **Mercury 3** | — | M/B | 🔒 API | 1M context, ~1,000 tok/s | [blog](https://www.inceptionlabs.ai/blog/introducing-mercury-3) |
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
| **LLaDA2.1 — Speeding Up Text Diffusion via Token Editing** | arXiv'26 | [abs](https://arxiv.org/abs/2602.08676) · [code](https://github.com/inclusionAI/LLaDA2.X) | Adds token-to-token editing to mask-to-token decoding in the 100B LLaDA2 models, with a fast "Speedy" mode and a higher-quality mode. |
| **iLLaDA — Improved Large Language Diffusion Models** | arXiv'26 | [abs](https://arxiv.org/abs/2606.25331) · [code](https://github.com/ML-GSAI/LLaDA) · [HF](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct) | 8B masked dLLM trained from scratch on 12T tokens (8K context); +21.6 BBH (base) and +16.5 HumanEval (instruct) over LLaDA. |
| **Sumi — Open Uniform Diffusion LM from Scratch** | arXiv'26 | [abs](https://arxiv.org/abs/2606.19005) | Fully open 7B **uniform-state** (not masked) diffusion LM pretrained on 1.5T tokens, with weights, checkpoints and recipe. |
| **LLaDA2.2 — Agentic dLLMs via Levenshtein Editing** | report'26 | [report](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf) · [HF](https://huggingface.co/inclusionAI/LLaDA2.2-flash) | 100B MoE dLLM (128K context) adding DELETE/INSERT control tokens so decoding can edit length, aimed at multi-turn tool use. |

### 3.2 Frontier / commercial dLLMs

| Model | Org | Links | TL;DR |
|---|---|---|---|
| **Mercury — Ultra-Fast Diffusion LMs** | Inception Labs | [abs](https://arxiv.org/abs/2506.17298) · [blog](https://www.inceptionlabs.ai/blog/introducing-mercury) · [demo](https://chat.inceptionlabs.ai) | First commercial-scale dLLM; Mercury Coder **1000+ tok/s** on H100 at frontier quality. |
| **Gemini Diffusion** | Google DeepMind | [page](https://deepmind.google/models/gemini-diffusion/) · [blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/) | Frontier-lab text diffusion (1000–2000 tok/s) matching Gemini 2.0 Flash-Lite on coding. |
| **Seed Diffusion** | ByteDance Seed | [abs](https://arxiv.org/abs/2508.02193) | Large-scale code dLLM at **2146 tok/s** (5.4× faster than comparable AR). |
| **Mercury 2 / 2.5 / 3** | Inception Labs | [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2) · [2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) · [3](https://www.inceptionlabs.ai/blog/introducing-mercury-3) | Feb 2026: first reasoning dLLM (~1,000 tok/s on Blackwell). Oct 2026: Mercury 3 with **1M-token context** at ~1,070 tok/s (vendor-reported). |
| **DiffusionGemma** | Google DeepMind | [abs](https://arxiv.org/abs/2608.00146) · [blog](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/) · [HF](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | First **open-weight** frontier-lab dLLM (Jun 2026, Apache-2.0): Gemma 4 MoE (25.2B total / 3.8B active) refining 256-token blocks in parallel. |

### 3.3 Hybrid & block diffusion (AR ↔ diffusion)

Interpolating between autoregression and diffusion to recover KV-cache & quality.

| Paper | Venue | Links | TL;DR |
|---|---|---|---|
| **BD3-LM: Block Diffusion** | ICLR'25 (Oral) | [abs](https://arxiv.org/abs/2503.09573) · [code](https://github.com/kuleshov-group/bd3lms) | Block-wise diffusion → KV-caching, variable length, tunable AR↔diffusion; SOTA among diffusion LMs. |
| **Nemotron-Labs-Diffusion (NVIDIA)** | arXiv'26 | [abs](https://arxiv.org/abs/2607.05722) · [HF](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion) | One 3B/8B/14B model that switches among AR, block-diffusion and self-speculation decoding (diffusion drafts, AR verifies). |
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

## 17. Masking, noise schedules & unmasking order

The two design choices that most define a diffusion LM's behavior: the **forward** side (how tokens are corrupted — noise/masking schedule, masking rate, curriculum) and the **reverse** side (which masked positions to fill next, how many per step, and whether committed tokens can be **remasked** and corrected). This section collects work dedicated to those choices.

**Already elsewhere in this list (core references):** [Informed Correctors for Discrete Diffusion](https://arxiv.org/abs/2407.21243) · [DDPD — Think While You Generate (Planned Denoising)](https://arxiv.org/abs/2410.06264) · [Train for the Worst, Plan for the Best (token ordering)](https://arxiv.org/abs/2502.06768) · [ReMDM — Remasking + inference-time scaling](https://arxiv.org/abs/2503.00307) · [EB-Sampler — Entropy-Bounded Unmasking](https://arxiv.org/abs/2505.24857) · [SlowFast Sampling — three golden principles](https://arxiv.org/abs/2506.10848) · [Dilated Scheduling (Plan for Speed)](https://arxiv.org/abs/2506.19037) · [Any-Order GPT as Masked Diffusion Model](https://arxiv.org/abs/2506.19935) · [WINO — Wide-In, Narrow-Out revokable decoding](https://arxiv.org/abs/2507.18578) · [Empirical Analysis of Decoding Biases in MDMs](https://arxiv.org/abs/2508.13021) · [Prophet — dLLMs Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982) · [Set Block Decoding](https://arxiv.org/abs/2509.04185) · [Masked Diffusion Models are Secretly Learned-Order AR](https://arxiv.org/abs/2511.19152) · [Learning Unmasking Policies for dLLMs](https://arxiv.org/abs/2512.09106) · [Mask Is What DLLM Needs (priority masking)](https://arxiv.org/abs/2603.15803) · [Understanding & Accelerating MDM Training (bell-shaped schedule)](https://arxiv.org/abs/2605.13026) · [Re-evaluating Confidence Remasking](https://arxiv.org/abs/2606.12232)

**Net-new from the focused gap-sweep (45 papers, individually link-verified), oldest→newest:**

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2022-05 | **Training and Inference on Any-Order Autoregressive Models the Right Way** | [abs](https://arxiv.org/abs/2205.13554) | Foundational theory paper on any-order autoregressive models that removes redundancy in order-agnostic training/inference, underpinning later masked-diffusion order-equivalence results. |
| 2025-02 | **Path Planning for Masked Diffusion Model Sampling** | [abs](https://arxiv.org/abs/2502.03540) | Decomposes each generation step into a planner (which positions to update, including remasking) and a denoiser, generalizing existing order heuristics. |
| 2025-08 | **The Cosine Schedule is Fisher-Rao-Optimal for Masked Discrete Diffusion Models** | [abs](https://arxiv.org/abs/2508.04884) | Uses information geometry to show the widely used cosine noise schedule is Fisher-Rao-optimal for masked discrete diffusion, giving theory for schedule choice. |
| 2025-08 | **MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2508.13148) | Analyzes the mismatch where training masks tokens randomly while inference reveals structured order, targeting the core train-vs-inference ordering discrepancy in masked diffusion LMs. |
| 2025-09 | **Masked Diffusion Language Models with Frequency-Informed Training** | [abs](https://arxiv.org/abs/2509.05056) | Compares linear/cosine noise schedules and NELBO weighting schemes, then adds frequency-informed curriculum masking that raises masking probability for rare tokens. |
| 2025-09 | **Planner Aware Path Learning in Diffusion Language Models Training** | [abs](https://arxiv.org/abs/2509.23405) | Introduces a planned ELBO (P-ELBO) so training aligns with planner-based inference that iteratively selects where to denoise along the sequence. |
| 2025-09 | **RIV: Recursive Introspection Mask Diffusion Vision Language Model** | [abs](https://arxiv.org/abs/2509.23625) | An introspection model detects errors in generated sequences and recursively alternates unmask, error-detection, and remask to self-correct committed tokens. |
| 2025-09 | **Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models** | [abs](https://arxiv.org/abs/2509.23653) | Introduces RemeDi, jointly predicting tokens and per-token confidence so already-revealed low-confidence tokens are remasked and resampled instead of being permanently fixed. |
| 2025-09 | **AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size** | [abs](https://arxiv.org/abs/2509.26432) | Training-free scheduler that adapts block size at runtime from confidence-volatility bands, directly targeting block-size selection for parallel decoding. |
| 2025-10 | **Fine-Tuning Masked Diffusion for Provable Self-Correction** | [abs](https://arxiv.org/abs/2510.01384) | PRISM adds a plug-in remasking self-correction loss that provably learns per-token quality scores to detect and revise committed tokens at inference without RL or a verifier. |
| 2025-10 | **Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies** | [abs](https://arxiv.org/abs/2510.05725) | Replaces max-confidence heuristics with a learned scheduler formulated as a KL-regularized MDP to decide which position to unmask next. |
| 2025-10 | **Soft-Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2510.17206) | Soft-masking blends mask and top-k predicted embeddings so partial information persists and revealed positions remain revisable across iterative refinement steps. |
| 2025-10 | **Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing** | [abs](https://arxiv.org/abs/2510.21961) | PUNT is a model-agnostic sampler that tests token conditional independence and drops conflicting low-confidence predictions to safely raise the number of tokens unmasked in parallel. |
| 2025-10 | **Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations** | [abs](https://arxiv.org/abs/2510.25544) | Derives dimension-free error bounds and identifies the optimal (non-constant) masking schedule as a function of the data's information profile for masked discrete diffusion. |
| 2025-11 | **Beyond Static Cutoffs: One-Shot Dynamic Thresholding for Diffusion Language Models** | [abs](https://arxiv.org/abs/2511.02077) | Calibrates confidence thresholds on a single sequence and reuses them to dynamically set how many tokens are committed per parallel decoding step. |
| 2025-11 | **Bringing Stability to Diffusion: Decomposing and Reducing Variance of Training Masked Diffusion Models** | [abs](https://arxiv.org/abs/2511.18159) | Decomposes MDM training variance into masking-pattern, masking-rate, and data noise and proposes a Pareto-optimal time-sampler that reshapes the training time-sampling distribution. |
| 2025-11 | **From Bits to Rounds: Parallel Decoding with Exploration for Diffusion Language Models** | [abs](https://arxiv.org/abs/2511.21103) | Explore-Then-Exploit combines cross-block decoding with targeted exploration of high-uncertainty tokens to maximize information throughput (tokens committed) per denoising round. |
| 2025-12 | **Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty** | [abs](https://arxiv.org/abs/2512.21336) | Formalizes Path Uncertainty and a Denoising Entropy metric to characterize and select optimal decoding orders in masked diffusion, giving a principled theory of which generation path to follow. |
| 2026-01 | **STaRR: Spatial-Temporal Token-Dynamics-Aware Responsive Remasking for Diffusion Language Models** | [abs](https://arxiv.org/abs/2601.04205) | Training-free responsive remasking that uses temporal-variance and spatial-deviance token dynamics to decide which committed tokens to revert to masks instead of static confidence thresholds. |
| 2026-01 | **Beyond Hard Masks: Progressive Token Evolution for Diffusion Language Models** | [abs](https://arxiv.org/abs/2601.07351) | EvoToken-DLM replaces binary masking with evolving soft token distributions enabling a revisable, progressive mask-to-token transition during iterative refinement. |
| 2026-01 | **Improving Diffusion Language Model Decoding through Joint Search in Generation Order and Token Space** | [abs](https://arxiv.org/abs/2601.20339) | Proposes Order-Token Search that beam-searches jointly over generation order and token values rather than committing to one decoding trajectory. |
| 2026-02 | **Generation Order and Parallel Decoding in Masked Diffusion Models: An Information-Theoretic Perspective** | [abs](https://arxiv.org/abs/2602.00286) | Provides a unified information-theoretic framework decoupling order sensitivity from parallelization bias, analyzing confidence/entropy/margin ordering heuristics. |
| 2026-02 | **Unifying Masked Diffusion Models with Various Generation Orders and Beyond** | [abs](https://arxiv.org/abs/2602.02112) | Treats the masking scheduler as a modeling component, replacing uniform random masking with order-induced corruption distributions for token/position-dependent forward masking. |
| 2026-02 | **Training-Free Self-Correction for Multimodal Masked Diffusion Models** | [abs](https://arxiv.org/abs/2602.02927) | Exploits pretrained MDM inductive biases to allow revision of earlier errors during sampling, making normally irreversible committed-token updates revocable without extra training. |
| 2026-02 | **CoRe: Context-Robust Remasking for Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.04096) | Identifies context-brittle tokens by probing sensitivity to masked-context perturbations to decide which positions to revise versus keep during decoding. |
| 2026-02 | **DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs** | [abs](https://arxiv.org/abs/2602.05992) | Uses dynamically sized sliding blocks adapted to semantic difficulty with a specialized KV-cache, choosing how large each denoising block should be. |
| 2026-02 | **Where-to-Unmask: Ground-Truth-Guided Unmasking Order Learning for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.09501) | Trains a supervised planning model with a Gt-Margin score to imitate an oracle unmasking order, directly learning which masked position to fill next. |
| 2026-02 | **Learn from Your Mistakes: Self-Correcting Masked Diffusion Models** | [abs](https://arxiv.org/abs/2602.11590) | Proposes Progressive Self-Correction (ProSeCo) training so the model both unmasks and iteratively corrects previously committed tokens during generation. |
| 2026-02 | **Why Any-Order Autoregressive Models Need Two-Stream Attention: A Structural-Semantic Tradeoff** | [abs](https://arxiv.org/abs/2602.16092) | Uses the masked-diffusion / any-order autoregressive equivalence to prove a structural-semantic tension that theoretically explains why arbitrary-order token prediction requires two-stream attention. |
| 2026-02 | **Adaptation to Intrinsic Dependence in Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.20126) | Proposes a distribution-agnostic, hyperparameter-free adaptive schedule for diffusion LMs that randomizes tokens revealed per step to match the target's dependence structure. |
| 2026-03 | **DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs** | [abs](https://arxiv.org/abs/2603.12996) | Builds a conditional dependency graph from self-attention and reduces per-step parallel unmasking to selecting an independent set, avoiding co-updating strongly coupled tokens. |
| 2026-03 | **DOS: Dependency-Oriented Sampler for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2603.15340) | Approximates inter-token dependencies from attention matrices to decide which masked positions can be safely unmasked together per step. |
| 2026-03 | **Confidence-Based Decoding is Provably Efficient for Diffusion Language Models** | [abs](https://arxiv.org/abs/2603.22248) | Gives the first formal efficiency guarantees for confidence- and entropy-sum-based unmasking order selection in masked diffusion sampling. |
| 2026-04 | **Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.02560) | DEMASK trains a lightweight dependency predictor and greedily selects positions with bounded cumulative dependency to unmask simultaneously, directly deciding how many tokens to commit per step. |
| 2026-04 | **Stability-Weighted Decoding for Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.17068) | Selects which tokens are safe to unmask using temporal KL stability between consecutive prediction distributions instead of raw confidence/entropy/margin. |
| 2026-04 | **DPRM: A Plug-in Doob h transform-induced Token-Ordering Module for Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.24357) | A plug-in module that modifies only the ordering policy via a Doob h-transform to choose token order without a fixed left-to-right constraint. |
| 2026-05 | **Edit-Based Refinement for Parallel Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.09603) | ME-DLM refines an initial parallel generation via replacement, deletion, and insertion edits conditioned on the full sequence, a backward self-correction pass over committed tokens. |
| 2026-05 | **Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement in Discrete Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.26436) | Training-free Token-to-Mask remasking resets suspected-wrong committed tokens back to [MASK] so the mask-filling steps re-predict them under cleaner context. |
| 2026-05 | **Cluster-Level Attention-Guided Parallel Decoding for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.29607) | Groups high-confidence predictions into clusters and uses attention to pick compatible clusters for simultaneous commitment, setting how many tokens unmask per step. |
| 2026-06 | **Revise, Don't Freeze: Sampler-Matched Training for Self-Correcting Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.01026) | Presents D3IM, a parameter-free sampler enabling direct visible-to-visible revision of committed tokens plus SCOPE training to overcome preservation bias. |
| 2026-06 | **Supportive Token Revealing for Fast Diffusion Language Model Decoding** | [abs](https://arxiv.org/abs/2606.04236) | AXON selects which confident reveals best support subsequent denoising rather than only the safest, changing the set and count of tokens unmasked in parallel. |
| 2026-06 | **NAVIRA: Decoupled Stochastic Remasking for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.06031) | Decouples token-quality assessment from regeneration and uses temperature-controlled stochastic remasking so early parallel-decoding errors can be revoked and re-decoded from cleaner context. |
| 2026-06 | **Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.10829) | ADAS reranks parallel unmasking candidates by discounting tokens that attend to uncertain just-committed positions, adaptively governing dependency-aware parallel commitment. |
| 2026-06 | **Neither Parallel Nor Sequential: How DiffusionGemma Actually Commits Tokens** | [abs](https://arxiv.org/abs/2606.14620) | Empirically analyzes the real token-commitment order of a large diffusion LLM, revealing a partial left-to-right bias between strict sequential and parallel unmasking. |
| 2026-06 | **Mean-Field Parallel Decoding for Discrete Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.15805) | Assigns commit scores to masked positions and refines them via pairwise interactions to coordinate how many tokens can be generated simultaneously per step. |

---

## 18. Extended index — deep gap-sweep (2022–2026)

An additional **125 papers** surfaced by an aggressive multi-agent gap-sweep and individually link-verified. Grouped by topic, oldest→newest. See [PAPERS.md](PAPERS.md) for the full sortable master table of every paper in this repo.

### 18 · 10. Guidance, control & constrained decoding

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

### 18 · 11. Applications

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

### 18 · 12. Safety, watermarking & robustness

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

### 18 · 15. "Language of X": biology & speech

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

### 18 · 16. Evaluation & benchmarks

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-02 | **Theoretical Benefit and Limitation of Diffusion Language Model** | [abs](https://arxiv.org/abs/2502.09622) | Analytically shows masked diffusion optimizes perplexity efficiently but needs sampling steps scaling linearly with sequence length to guarantee sequence-level correctness. |
| 2025-08 | **Empirical Analysis of Decoding Biases in Masked Diffusion Models** | [abs](https://arxiv.org/abs/2508.13021) · [code](https://github.com/NEUIR/Uncode) | ACL'26 (Oral) study identifying rigid-boundary and trivial-token decoding biases in uncertainty-based samplers, and proposing UNCODE, a training-free decoding calibration (positional-trajectory + content-informativeness priors) that beats existing decoding strategies by >7% on 7 reasoning/planning benchmarks across 3 MDMs and keeps the 2–4× speedups of fast samplers. |
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

### 18 · 2. Early & continuous text diffusion

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

### 18 · 4. Scaling laws & training

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

### 18 · 7. RL, post-training & alignment

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

### 18 · 9. Inference & acceleration

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

## 19. Oct 2026 update — Jul–Oct 2026 refresh

**268 papers** added in the October 2026 refresh (mostly arXiv 2606–2610, plus missing 2025–early-2026 work), found by six parallel search agents and individually checked against arXiv. Grouped by topic, oldest→newest. Headline models also appear in [§3](#3-large-scale-diffusion-llms-dllms) and [Models at a glance](#models-at-a-glance).

### 19 · Surveys & overviews

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-01 | **Top 10 Open Challenges Steering the Future of Diffusion Language Model and Its Variants** | [abs](https://arxiv.org/abs/2601.14041) | A perspective paper listing ten open challenges (architecture, tokenization, algorithms, reasoning, multimodality) and arguing for a shift from AR adaptation to a diffusion-native ecosystem. |
| 2026-07 | **Accelerating Masked Diffusion Large Language Models: A Survey of Efficient Inference Techniques** (IJCAI'26) | [abs](https://arxiv.org/abs/2607.12829) | Surveys dLLM acceleration through a unified latency-decomposition framework covering algorithmic, architectural/system, and inference-time-scaling techniques, with reproducible-benchmarking guidelines. |
| 2026-07 | **Discrete Diffusion Models: A Unified Framework from Tokenization to Generation** | [abs](https://arxiv.org/abs/2607.13431) | A conceptual framework that views discrete diffusion through state-space construction (tokenization, vocabulary topology, structural alphabets) and covers objectives, inference, scaling, systems, and evaluation. |
| 2026-09 | **Diffusion Language Models for Mobile Edge Agentic AI: Foundations, Applications, and Challenges** | [abs](https://arxiv.org/abs/2609.04778) | An overview of DLM properties (bidirectionality, parallel refinement, quality-latency elasticity) and how they map to mobile-edge agentic AI system requirements. |

### 19 · 1. Foundations & theory

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-11 | **Optimal Inference Schedules for Masked Diffusion Models** | [abs](https://arxiv.org/abs/2511.04647) | Exactly characterizes the divergence between the sampled and true distribution for any unmasking schedule, and shows that total-correlation-based schedules can sample in O(log n) steps in natural settings. |
| 2026-01 | **Corrected Samplers for Discrete Flow Models** | [abs](https://arxiv.org/abs/2601.22519) · [code](https://github.com/WanZhengyan/Corrected-Samplers-for-Discrete-Flow-Models) | Gives non-asymptotic error bounds for tau-leaping and Euler samplers of discrete flow models, and proposes time- and location-corrected samplers that reduce discretization error at almost no extra cost. |
| 2026-02 | **The Diffusion Duality, Chapter II: Ψ-Samplers** | [abs](https://arxiv.org/abs/2602.21185) · [project](https://s-sahoo.com/duo-ch2) | Introduces predictor-corrector Ψ-samplers for discrete diffusion with arbitrary noise processes; with uniform-state diffusion they beat ancestral sampling and keep improving as steps increase. |
| 2026-02 | **Sharp Convergence Rates for Masked Diffusion Models** | [abs](https://arxiv.org/abs/2602.22505) | Derives tighter TV-distance convergence bounds for masked diffusion samplers under weaker assumptions, with a matching lower bound for Euler, and shows the First-Hitting Sampler adds no error beyond score estimation. |
| 2026-05 | **Discrete Flow Matching: Convergence Guarantees Under Minimal Assumptions** | [abs](https://arxiv.org/abs/2605.08882) | Proves non-asymptotic KL/TV convergence bounds for time-discretized discrete flow matching that need only an approximation-error assumption, with improved dependence on vocabulary size and dimension. |
| 2026-05 | **Vocabulary-size-independent Convergence of Discrete Diffusion Models: adjoint equations induce the right space** | [abs](https://arxiv.org/abs/2605.17232) | Uses an adjoint-equation framework to give vocabulary-size-independent convergence bounds in any integral probability metric for both masked and uniform priors. |
| 2026-07 | **Context-weighted Discrete Flow Matching** | [abs](https://arxiv.org/abs/2607.21427) | Links per-token uncertainty to the density of surrounding context and proposes a context-weighted sampler and scaled cross-entropy loss for DFM, reporting up to 63% lower OpenWebText generative perplexity. |
| 2026-08 | **Minimax Optimality of Score-Entropy Discrete Diffusion** | [abs](https://arxiv.org/abs/2608.20635) | Derives a minimax lower bound for estimating concrete scores in uniform and masking discrete diffusion, and gives a thresholded MLE that nearly matches it, implying SEDD can be near-optimal in KL sample complexity. |
| 2026-08 | **Provably adaptive sampling with uniform and remasking discrete diffusion models** | [abs](https://arxiv.org/abs/2608.23554) | Shows that a leave-one-out-denoiser sampler for uniform and remasking discrete diffusion needs a number of steps governed by the data's dual total correlation rather than the ambient dimension. |
| 2026-08 | **The information geometry of product-reference discrete diffusion: Interaction growth complexity and optimal scheduling** | [abs](https://arxiv.org/abs/2608.28949) | Defines interaction growth complexity for product-reference discrete diffusion samplers and shows how it governs KL discretization error, stepsize schedules, and the choice of reference distribution. |
| 2026-09 | **Parallelism, critical windows, and separations among diffusion language models** | [abs](https://arxiv.org/abs/2609.20539) | Compares the parallel sampling capacity of masked, uniform, and Gaussian diffusion, proving that uniform and Gaussian diffusion can need far fewer forward passes on some distributions because of masked diffusion's narrower critical windows. |
| 2026-09 | **Does Uniform Discrete Diffusion Need Time?** | [abs](https://arxiv.org/abs/2609.30977) | Argues that the optimal uniform-diffusion predictor's time dependence becomes negligible with finite language data, and shows that time-agnostic models match or beat time-conditioned ones. |
| 2026-10 | **Uniform Discrete Diffusion Models are Minimax Optimal for Estimating Distributions with Small Effective Support Size** | [abs](https://arxiv.org/abs/2610.07655) | Proves that uniform discrete diffusion achieves TV error scaling with the effective support size, minimax optimal in TV and optimal up to log n in KL. |
| 2026-10 | **Transformer-Based Masked Diffusion Models Provably Learn Distributions with Latent Structure** | [abs](https://arxiv.org/abs/2610.12544) | Gives end-to-end statistical guarantees that transformer-based MDMs learn mixtures of hidden Markov trees with sample complexity set by latent structure rather than the full sequence space. |
| 2026-10 | **Score Mismatch and Set-Constrained Sampling in Zero-Shot Conditional Discrete Diffusion** | [abs](https://arxiv.org/abs/2610.21061) | Derives a KL bound that isolates the bias of reusing an unconditional discrete diffusion model for conditional targets, and proposes a Randomized Hard Clamping sampler for coordinatewise set constraints. |
| 2026-10 | **Diffusion Language Models as Open Quantum Systems** | [abs](https://arxiv.org/abs/2610.21672) | Maps the forward and reverse dLLM dynamics onto open quantum systems theory and derives superposed-prompt generation and a "Quantum-Inspired Remasking" revision rule. |
| 2026-10 | **Query-Efficient Rejection Sampling for Parallel Masked Diffusion** | [abs](https://arxiv.org/abs/2610.22238) | Proposes SIRS, a rejection sampler that corrects parallel-unmasking bias with one random-token score query per step, and bounds output KL by a clipping term plus training error. |

### 19 · 2. Early & continuous text diffusion

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-05 | **Smoothie: Smoothing Diffusion on Token Embeddings for Text Generation** | [abs](https://arxiv.org/abs/2505.18853) · [code](https://github.com/ashaba1in/smoothie) | Diffuses by progressively smoothing token embeddings according to semantic similarity, and beats prior diffusion models on seq2seq and unconditional generation. |
| 2025-10 | **Continuously Augmented Discrete Diffusion model for Categorical Generative Modeling** | [abs](https://arxiv.org/abs/2510.01329) | CADD pairs masked discrete diffusion with a continuous latent diffusion so that masked positions carry informative latents, improving text, image, and code generation. |
| 2026-02 | **Flow Map Language Models: One-step Language Modeling via Continuous Denoising** | [abs](https://arxiv.org/abs/2602.16813) · [code](https://github.com/david3684/flm) | Continuous flows over token representations beat discrete diffusion in quality and speed, and a distilled flow map (FMLM) generates in one step with better quality than 8-step discrete diffusion. |
| 2026-04 | **LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling** | [abs](https://arxiv.org/abs/2604.11748) · [code](https://github.com/nealchen2003/LangFlow) | Connects embedding-space diffusion to flow matching via Bregman divergence, with an ODE NLL bound and a learnable noise schedule, matching top discrete DLMs on PPL and GenPPL. |
| 2026-05 | **Continuous Latent Diffusion Language Model** | [abs](https://arxiv.org/abs/2605.06548) | Cola DLM: a Text VAE, a block-causal DiT prior in continuous latent space, and a conditional decoder, with strong scaling against matched ~2B AR and LLaDA baselines (project page hongcanguo.github.io/Cola-DLM). |
| 2026-05 | **TextLDM: Language Modeling with Continuous Latent Diffusion** | [abs](https://arxiv.org/abs/2605.07748) | Applies the image DiT recipe to text, using a representation-aligned VAE latent and flow matching, and matches GPT-2 under equal training settings. |
| 2026-05 | **ELF: Embedded Language Flows** | [abs](https://arxiv.org/abs/2605.10938) | Flow matching in a continuous embedding space, discretized only at the final step with a shared-weight network, outperforms leading discrete and continuous DLMs with fewer steps. |
| 2026-05 | **Language Modeling with Hyperspherical Flows** | [abs](https://arxiv.org/abs/2605.11125) | S-FLM generates text by rotating vectors on a hypersphere along a cross-entropy-learned velocity field, nearly matching masked diffusion at standard temperature. |
| 2026-08 | **ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings** | [abs](https://arxiv.org/abs/2608.23551) · [code](https://github.com/Na-Li66/ConvergeFlow) | Restricts the data predictor to the convex hull of token embeddings and proves the flow converges to valid embeddings, removing the need for a cross-entropy decoder. |
| 2026-09 | **ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks** | [abs](https://arxiv.org/abs/2609.29102) | Extends continuous DLMs to GSM8K, MATH-500, HumanEval, and MBPP by using a frozen AR teacher to guide intermediate denoiser features. |
| 2026-09 | **Simplex Diffusion Models** | [abs](https://arxiv.org/abs/2609.35553) | Runs diffusion over categorical distributions on the probability simplex to keep uncertainty between steps, and is competitive on text, code, and math, including after distillation to 8 steps. |
| 2026-09 | **Distribution Matching Distillation for Continuous Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.40235) | Proposes Simplex-DMD and Reinforce-DMD distillation for continuous DLMs; Simplex-DMD reaches 45.6 GenPPL at 4 NFEs on OpenWebText. |
| 2026-10 | **Scaling and Distilling Text Embeddings for Better Diffusibility** | [abs](https://arxiv.org/abs/2610.01016) · [code](https://github.com/la0ka1/diffusing-scaled-text-embeddings) | Finds that larger frozen text encoders improve continuous DLMs, but overly discriminative embeddings hurt sampling, so it distills them into a smoother student latent space that beats GPT-2-M GenPPL. |
| 2026-10 | **Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models** (TMLR'26) | [abs](https://arxiv.org/abs/2610.09145) · [code](https://github.com/LateralIntelligence/noise-your-prompt) | Shows that noising prompt tokens during training improves generalization of continuous DLMs on Sudoku and N-Queens, with modest gains on Gigaword. |
| 2026-10 | **Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization** | [abs](https://arxiv.org/abs/2610.09311) | BLD compresses 1024 tokens into 64 block latents decoded by parallel local AR branches, cutting generation FLOPs relative to ELF-L and AR baselines. |
| 2026-10 | **Efficient Reasoning with Flow Language Models** | [abs](https://arxiv.org/abs/2610.09416) | Finds that continuous-state flow LMs reach higher few-step accuracy than discrete diffusion on maze and Sudoku, and match baselines with fewer parameters. |
| 2026-10 | **One-Step Text Generation by Seq-Drifting** | [abs](https://arxiv.org/abs/2610.12879) · [code](https://github.com/jyliu-ai/Seq-Drifting) | Learns a direct Gaussian-noise-to-text map for single-pass sequence generation, reaching 36.01 GenPPL on LM1B in one step. |
| 2026-10 | **Rethinking the Roles of Diffusion and Decoding: Predictive-State Flows for Parallel Language Generation** | [abs](https://arxiv.org/abs/2610.12898) | A diffusion prior generates compact block-level predictive states, which a decoder turns into tokens (parallel across blocks, AR within each block), beating ELF-B GenPPL at a fraction of the compute. |
| 2026-10 | **On the Design Space of Continuous Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.13834) | Studies self-conditioning and embedding geometry, and proposes DoubleCat/SemiCat with discrete reveals to fix token identifiability; SemiCat reaches discrete-diffusion level on TinyGSM. |
| 2026-10 | **Designing Transformers for Continuous Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.22330) | LangDiT adds normalization changes, wider noise conditioning, gradient routing, and self-conditioning, lowering OpenWebText GenPPL relative to LangFlow and Plaid-style backbones. |

### 19 · 3. Large-scale diffusion LLMs (dLLMs)

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-12 | **Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed** (ICML'26) | [abs](https://arxiv.org/abs/2512.14067) | An AR-to-dLM conversion recipe with block-wise attention (causal across blocks, bidirectional within) that preserves AR weight distributions; the 8B model beats Dream-7B and Qwen3-4B in accuracy and throughput. |
| 2025-12 | **WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference** (ICML'26) | [abs](https://arxiv.org/abs/2512.22737) · [code](https://github.com/Tencent/WeDLM) | A diffusion LM that keeps a strict causal mask via topological reordering of observed tokens, enabling prefix KV caching and up to ~3x (reasoning) / 10x (low-entropy) speedups over vLLM-served AR models. |
| 2026-01 | **Diffusion In Diffusion: Reclaiming Global Coherence in Semi-Autoregressive Diffusion** *(hybrid/block)* | [abs](https://arxiv.org/abs/2601.13599) | A draft-then-refine framework that restores global bidirectional coherence on top of block diffusion, cutting generative perplexity from 25.7 to 21.9 with 26% of the fine-tuning budget. |
| 2026-01 | **Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model** | [abs](https://arxiv.org/abs/2601.15892) · [code](https://github.com/ByteDance-Seed/Stable-DiffCoder) | A block-diffusion code model built by continual pretraining on the Seed-Coder architecture, data, and pipeline, with a tailored warmup and block-wise clipped noise schedule; it outperforms its AR counterpart under matched data. |
| 2026-01 | **Causal Autoregressive Diffusion Language Model (CARD)** *(hybrid/block)* | [abs](https://arxiv.org/abs/2601.22031) | Reformulates diffusion under a strictly causal mask for dense per-token supervision in one forward pass, with soft-tailed masking and SNR-based reweighting for stability. |
| 2026-02 | **LLaDA2.1: Speeding Up Text Diffusion via Token Editing** | [abs](https://arxiv.org/abs/2602.08676) · [code](https://github.com/inclusionAI/LLaDA2.X) | Adds Token-to-Token editing to Mask-to-Token decoding in the 100B LLaDA2 block-diffusion models, giving a configurable threshold scheme with a fast "Speedy" mode and a higher-quality mode. |
| 2026-04 | **From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons (FLUID)** (ACL'26) | [abs](https://arxiv.org/abs/2605.27387) | Adapts GPT-style AR checkpoints to diffusion via Strictly Causal Alignment plus an entropy-driven Elastic Horizons denoising schedule, avoiding pretraining from scratch. |
| 2026-05 | **Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment** | [abs](https://arxiv.org/abs/2605.06885) · [code](https://github.com/pengzhangzhi/Open-dLLM) | Converts AR models to masked DLMs by explicitly aligning the DLM's internal representations to those of the same-architecture AR model rather than relying only on continued denoising training. |
| 2026-06 | **FLARE: Diffusion for Hybrid Language Model** *(hybrid/block)* | [abs](https://arxiv.org/abs/2606.01774) | Converts hybrid-attention AR LLMs into block-diffusion LMs, combining efficient per-call architectures with parallel denoising, and analyzes transfer-data and objective bottlenecks. |
| 2026-06 | **BlockGen: Flexible Blockwise Sequence Modeling with Hybrid Samplers** *(hybrid/block)* | [abs](https://arxiv.org/abs/2606.02241) · [code](https://github.com/jdeschena/blockgen) | A blockwise framework comparing uniform-state and masked diffusion under informed predictor-corrector samplers, finding that the USDM advantage closes or reverses at high NFE. |
| 2026-06 | **Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation (OPDLM)** | [abs](https://arxiv.org/abs/2606.06712) | Converts AR LMs into DLMs by on-policy self-distillation from a frozen AR teacher on the student's own trajectories, reducing knowledge loss and the train-inference mismatch. |
| 2026-06 | **Sumi: Open Uniform Diffusion Language Model from Scratch** | [abs](https://arxiv.org/abs/2606.19005) | A fully open 7B uniform-state diffusion LM pretrained from scratch on 1.5T tokens, with weights, checkpoints, and recipe released as a reference point for uniform diffusion at scale. |
| 2026-06 | **Improved Large Language Diffusion Models (iLLaDA)** | [abs](https://arxiv.org/abs/2606.25331) · [code](https://github.com/ML-GSAI/LLaDA) | An 8B fully bidirectional masked diffusion LM trained from scratch on 12T tokens plus 25B-token SFT; it improves on LLaDA (e.g., +21.6 BBH base, +16.5 HumanEval instruct) and is competitive with Qwen2.5-7B on several benchmarks. |
| 2026-06 | **Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context** *(hybrid/block)* | [abs](https://arxiv.org/abs/2606.26493) · [HF](https://huggingface.co/collections/nvidia/nemotron-labs-twotower) | A block-wise AR-diffusion model with a frozen causal AR context tower and a trainable bidirectional block-diffusion denoiser tower that cross-attends to it. |
| 2026-06 | **Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation** *(hybrid/block)* | [abs](https://arxiv.org/abs/2606.27732) | An asymmetric attention design that keeps KV caching while giving masked positions bidirectional context, reporting 1.9–2.9x speedups over AR baselines in batch serving. |
| 2026-06 | **Multi-Block Diffusion Language Models** *(hybrid/block)* | [abs](https://arxiv.org/abs/2606.29215) | Extends block diffusion to decode several consecutive blocks concurrently; MBD-LLaDA2-Mini raises tokens-per-forward from 3.47 to 6.19 with slightly higher accuracy. |
| 2026-06 | **Adaptive Block Diffusion: Resolving Training-Inference Mismatch in Diffusion Language Models** *(hybrid/block)* | [abs](https://arxiv.org/abs/2606.29275) | Trains one model over a distribution of prefix-window (block) configurations, removing off-grid collapse and matching fixed-block specialists at their own block sizes. |
| 2026-07 | **Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding** (ICML'26) *(hybrid/block)* | [abs](https://arxiv.org/abs/2607.01775) · [code](https://github.com/kuleshov-group/setdlms) | A likelihood parameterization over flexible-position, flexible-size token sets that generalizes block diffusion, supporting KV caching, variable-length generation, and stronger infilling. |
| 2026-07 | **Training Hybrid Block Diffusion Language Models with Partial Bidirectionality** *(hybrid/block)* | [abs](https://arxiv.org/abs/2607.02805) | Trains block-diffusion LMs on Mamba/attention hybrid backbones; BDLM Mamba-H reaches 19.7x the throughput of DiffuMamba-H at 65K tokens. |
| 2026-07 | **Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding** *(hybrid/block)* | [abs](https://arxiv.org/abs/2607.05722) · [HF](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B) | A single model (3B/8B/14B) trained with a joint AR-diffusion objective that switches among AR, block-diffusion, and self-speculation (diffusion drafts, AR verifies) decoding modes. |
| 2026-07 | **UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective** | [abs](https://arxiv.org/abs/2607.24507) | Adapts AR checkpoints directly to uniform-noise diffusion by unifying SEDD, MDLM/GIDD, M2S, and Neural CTMC losses as one generalized KL objective over reverse rates. |
| 2026-07 | **DiffusionGemma Technical Report** | [abs](https://arxiv.org/abs/2608.00146) | An experimental open-weight dLLM fine-tuned from MoE Gemma 4 (3.8B active / 25.2B total) that refines 256-token blocks in parallel; announced June 10, 2026 under Apache-2.0 (blog: https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/). |
| 2026-09 | **Unlocking Lossless Speedups in LLMs via Discrete Diffusion (Uno)** *(hybrid/block)* | [abs](https://arxiv.org/abs/2609.04010) · [project](https://s-sahoo.github.io/uno/) | "Diffusion-augmented LLMs" split parameters into NTP-trained AR weights and lightweight diffusion weights that draw multiple tokens from the AR distribution, giving up to 3x lossless speedups. |
| 2026-09 | **Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale** *(hybrid/block)* | [abs](https://arxiv.org/abs/2609.11998) | Shows that linear-state sequence mixers (Mamba, hybrids) give block diffusion an exact constant-size block cache and retrieve out to 8–16x training length, where attention collapses at 2x. |
| 2026-09 | **Zarya: A Hybrid Autoregressive–Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference** *(hybrid/block)* | [abs](https://arxiv.org/abs/2609.19868) | Jointly optimizes AR and masked-diffusion objectives in one architecture with variable-size slots, supporting both diffusion sampling and slotted speculative decoding; released at 0.6B/1.7B/4B. |
| 2026-09 | **dQwen3.5: Hybrid-Attention Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.20751) | Adapts hybrid attention+RNN Qwen3.5 backbones (0.8B–9B) into diffusion LMs, showing that structurally causal hybrid architectures can be converted into effective DLMs. |
| 2026-09 | **DreamingGoose: Staged Distillation from Autoregressive Transformers to Bidirectional Recurrent Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.34253) · [code](https://github.com/JIBSIL/dualgoose) | Converts Qwen3 1.7B/8B teachers into attention-free bidirectional gated-delta-rule diffusion students in three stages, finding that language modeling transfers partially and in-context retrieval does not. |
| 2026-09 | **d-OPD: Future-Aware On-Policy Distillation for Block Diffusion Language Models** *(hybrid/block)* | [abs](https://arxiv.org/abs/2609.35362) | Future-aware on-policy distillation for adapting pretrained AR models into block-diffusion LMs. |
| 2026-10 | **Context-Tower Conversion Preserves Generation While Freezing Retains Knowledge: Low-Budget AR-to-Diffusion Conversion of MoE LLMs** | [abs](https://arxiv.org/abs/2610.02657) | Compares in-place vs. context-tower AR-to-dLLM conversion of the same 30B MoE parent under a fixed corpus, token budget, and evaluation harness. |
| 2026-10 | **Large Language Continuous Diffusion Models (Sigma)** | [abs](https://arxiv.org/abs/2610.02665) | Sigma is the first large-scale (3B/8B) continuous diffusion LM; it is trained blockwise by likelihood on Gaussian-corrupted embeddings and warm-started from AR weights. |

### 19 · 4. Scaling laws & training

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-12 | **Dual-objective Language Models: Training Efficiency Without Overfitting** (ICLR'26) | [abs](https://arxiv.org/abs/2512.14549) | Mixing AR and masked-diffusion objectives without architecture changes outperforms either alone under data repetition, and the optimal mix is similar for AR and diffusion downstream use. |
| 2026-02 | **dLLM: Simple Diffusion Language Modeling** | [abs](https://arxiv.org/abs/2602.22661) · [code](https://github.com/ZHZisZZ/dllm) | An open-source framework that unifies training (masked and block diffusion), plug-in decoding, and reproducible evaluation for LLaDA/Dream-style DLMs, with small from-scratch and BERT/AR-converted DLM recipes. |
| 2026-03 | **MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Scaling of Diffusion Language Models** (EMNLP'26) | [abs](https://arxiv.org/abs/2603.16077) · [code](https://github.com/chen-hao-chao/mdm-prime-v2) | Fixes the MDM-Prime sub-tokenizer with binary encoding and index shuffling and provides token-granularity guidance, scaling partial-masking MDMs to 1.1B parameters that beat similar-size AR baselines on zero-shot commonsense tasks. |
| 2026-05 | **Continuous Diffusion Scales Competitively with Discrete Diffusion for Language** | [abs](https://arxiv.org/abs/2605.18530) | RePlaid aligns Plaid with modern discrete DLM architectures and gives the first continuous-DLM scaling law competitive with discrete DLMs (about a 20x compute gap to AR). |
| 2026-08 | **LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.03457) | A systematic scaling study of MoE dLLMs: the optimal batch size grows faster and the optimal LR decays faster with compute than in AR; IsoFLOP analysis also gives model-data allocation results. |
| 2026-08 | **Reducing Pretraining-Generation Mismatch in Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.09424) | Prefix-Conditioned Diffusion (PCD) samples a boundary, keeps the prefix clean with AR supervision, and denoises only the suffix during the stable stage of AR-to-block-diffusion conversion. |
| 2026-09 | **Less Uniform Discrete Diffusion is More Powerful and Scalable** | [abs](https://arxiv.org/abs/2609.35817) | LUDI fixes over-uniform objectives and condition-target confusion in uniform diffusion LMs with a less-uniform loss and per-token time embeddings, reaching 3 tokens/step with complex reasoning. |
| 2026-10 | **Revisiting the 16x Pretraining Compute Gap between Diffusion and Autoregressive Language Models** | [abs](https://arxiv.org/abs/2610.17590) · [project](https://guanghanwang.com/dlm-scaling) | Argues that the widely cited 16x DLM-vs-AR compute gap is a methodological artifact of ELBO scoring; with exact likelihood, block diffusion, and an encoder-decoder design, the gap shrinks to about 3x at small block sizes. |

### 19 · 5. Reasoning

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-08 | **Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models** | [abs](https://arxiv.org/abs/2508.09138) | Identifies "temporal oscillation" (correct answers appear mid-denoising then get overwritten) and proposes temporal voting plus a temporal-consistency RL reward. |
| 2025-10 | **Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts** | [abs](https://arxiv.org/abs/2510.05040) | HEX majority-votes across varied block-size schedules, exploiting an implicit ensemble of generation orders in dLLMs for training-free reasoning gains. |
| 2025-10 | **On the Reasoning Abilities of Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2510.13117) | Proves MDMs are equivalent to padded looped transformers (and related to CoT transformers) in a finite-precision log-width setting, and more efficient on some problems such as regular languages. |
| 2025-10 | **Diffuse Thinking: Exploring Diffusion Language Models as Efficient Thought Proposers for Reasoning** (ACL'26) | [abs](https://arxiv.org/abs/2510.27469) · [project](https://anonymous.4open.science/r/Diffuse-Thinking-EC60) | Uses DLMs to propose candidate intermediate thoughts in parallel and an LLM to evaluate them, reducing reasoning compute versus AR proposers. |
| 2026-01 | **Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models** | [abs](https://arxiv.org/abs/2601.22035) | Masked DLMs stay accurate under answer-first prompting (order robustness) unlike AR models, because simpler reasoning tokens stabilize earlier than final answers. |
| 2026-02 | **Reasoning with Latent Tokens in Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.03769) | Attributes DLM reasoning gains to joint prediction of undecoded "latent" tokens, shows their count trades speed for quality, and transfers the idea to AR models via multi-token prediction. |
| 2026-02 | **Adaptive Test-Time Compute Allocation for Block Diffusion Language Models in Complex Reasoning** | [abs](https://arxiv.org/abs/2602.09555) | Allocates compute adaptively at decoding and block-generation levels for block DLMs; the resulting TDAR-8B is faster and more accurate than TraDo-8B. |
| 2026-02 | **Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching** (NeurIPS'26) | [abs](https://arxiv.org/abs/2602.22871) · [code](https://github.com/roymiles/diffusion-stitching) | Samples many cheap masked-DLM reasoning traces, scores steps with a PRM, and stitches the best steps into a rationale for an AR solver, improving math/coding accuracy at lower latency. |
| 2026-03 | **Masked diffusion LLMs can use EoS tokens for hidden reasoning** | [abs](https://arxiv.org/abs/2603.05197) | Shows masked dLLMs use hidden states of padding EoS tokens as extra computation, improving arithmetic, entity-tracking, and Sudoku performance. |
| 2026-04 | **$S^3$: Stratified Scaling Search for Test-Time in Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.06260) | Expands candidates at each denoising step, scores them with a lightweight verifier, and resamples while keeping diversity, giving test-time gains on LLaDA-8B-Instruct especially for math. |
| 2026-04 | **Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models** (ICML'26) | [abs](https://arxiv.org/abs/2604.16565) | A training-free forward-mask/backward-reconstruct consistency metric that verifies dLLM reasoning, guides resampling, and serves as an alignment reward. |
| 2026-05 | **Looped Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.26106) | LoopMDM repeats early-middle transformer layers of masked DLMs to improve training efficiency and reasoning, with inference compute scaled by the number of loops. |
| 2026-06 | **Recursive Scaling in Masked Diffusion Models** | [abs](https://arxiv.org/abs/2606.18022) | Recursive MDMs reuse one denoising transformer several times per step to add depth without parameters, improving quality and cutting denoising steps on Sudoku and Countdown. |
| 2026-06 | **Scheduling Thoughts: Learning the Order of Thought in Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.23567) | Self-Aware Scheduling treats the unmasking order of a masked DLM as a learnable policy trained with a self-likelihood reward, improving puzzle (Sudoku) and math reasoning over heuristic orders. |
| 2026-08 | **Prefix-Denoising Consistency: Test-Time Verification for Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.25311) | PDC splits a sample at an intermediate point, regenerates the suffix from the fixed prefix, and uses the stability of the results as a correctness signal for self-verification on math and commonsense reasoning. |
| 2026-09 | **Register Tokens for Bounded-State Reasoning in Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.16372) | Carries reasoning progress between generation chunks of masked DLMs through a few trained register tokens instead of keeping earlier text in context, beating discrete-text carry on LLaDA and Dream. |

### 19 · 6. Coding & agents

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-08 | **TreeDiff: AST-Guided Code Generation with Diffusion LLMs** | [abs](https://arxiv.org/abs/2508.01473) | Masks tokens tied to key AST nodes instead of random tokens when training code dLLMs, reporting a 13.3% relative gain over random masking. |
| 2026-05 | **Beyond Execution: Static-Analysis Rewards and Hint-Conditioned Diffusion RL for Code Generation** | [abs](https://arxiv.org/abs/2605.17174) | Studies reward design, hint-conditioned sampling, and task difficulty in RL post-training of diffusion code models, finding static checking the strongest execution-free reward. |
| 2026-07 | **Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL** (NeurIPS'26) | [abs](https://arxiv.org/abs/2607.16204) · [code](https://github.com/patronus-ai/mdlm_world_modeling) | Uses MDLMs anchored on tool schemas as text-based world models for agent environments, outperforming much larger AR simulators, with GRPO on simulated rollouts transferring zero-shot to unseen environments. |
| 2026-10 | **Plan-and-Patch: Diffusion Language Models for Agentic Planning** | [abs](https://arxiv.org/abs/2610.10786) | A DLM generates structured agent plans and repairs only faulty segments while keeping other steps fixed, beating an AR baseline on plan repair and cutting planning latency on ALFWorld and TextCraft. |
| 2026-10 | **Diff Diffusion: Predicting Edit Paths with Diffusion Models for Efficient Code Editing** (NeurIPS'26 Workshop) | [abs](https://arxiv.org/abs/2610.17831) | Models code edits as structured operations decoded in parallel, matching AR baselines on edit-completion benchmarks at about 2.78x faster inference. |

### 19 · 7. RL, post-training & alignment

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-05 | **Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models** (NeurIPS'25) | [abs](https://arxiv.org/abs/2505.10446) · [code](https://github.com/maple-research-lab/LLaDOU) | DCoLT treats each intermediate denoising step as a latent reasoning action and applies outcome-based RL, improving math and code in dLLMs. |
| 2025-09 | **Inpainting-Guided Policy Optimization for Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2509.10396) | IGPO inpaints partial ground-truth reasoning during sampling to restore learning signal when GRPO groups would yield zero advantage. |
| 2025-10 | **Advancing Reasoning in Diffusion Language Models with Denoising Process Rewards** | [abs](https://arxiv.org/abs/2510.01544) | A process-level reward over the denoising trajectory that credits intermediate steps contributing to correct answers. |
| 2025-10 | **Simple Policy Gradients for Reasoning with Diffusion Language Models** (ICML'26) | [abs](https://arxiv.org/abs/2510.04019) · [code](https://github.com/probablyabot/agrpo) | AGRPO optimizes individual denoising steps rather than whole sequences, giving large gains over base LLaDA on Countdown and Sudoku. |
| 2025-10 | **SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models** (ICLR'26) | [abs](https://arxiv.org/abs/2510.09541) | Uses both upper and lower likelihood bounds for policy gradients in MDLMs, beating ELBO-based and one-step baselines on GSM8K, MATH500, Countdown, and Sudoku. |
| 2025-12 | **d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models** (ACL'26) | [abs](https://arxiv.org/abs/2512.09675) | Tree-structured rollouts with bottom-up verifiable-reward advantages plus a time-scheduled self-distillation loss for dLLM RL. |
| 2025-12 | **dUltra: Ultra-Fast Diffusion Language Models via Reinforcement Learning** | [abs](https://arxiv.org/abs/2512.21446) · [code](https://github.com/chinsengi/dUltra-os) | On-policy RL that teaches MDLMs to unmask many tokens per step, improving speed-accuracy trade-offs on math and code over heuristic and distillation baselines. |
| 2025-12 | **DiRL: An Efficient Post-Training Framework for Diffusion Language Models** | [abs](https://arxiv.org/abs/2512.22234) | A blockwise dLLM post-training framework (SFT + RL) introducing the GRPO-based DiPO and a math-focused DiRL-8B-Instruct. |
| 2026-01 | **T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforcement Learning** | [abs](https://arxiv.org/abs/2601.11214) | A TraceRL-based curriculum that progressively enlarges block size from an AR-initialized model, enabling more parallel decoding with little math accuracy loss. |
| 2026-01 | **The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models** (ICML'26) | [abs](https://arxiv.org/abs/2601.15165) · [code](https://github.com/LeapLabTHU/JustGRPO) | Argues arbitrary-order generation narrows dLLM reasoning and proposes JustGRPO (standard GRPO without arbitrary order), reaching 89.1% GSM8K while keeping parallel decoding. |
| 2026-02 | **LaDi-RL: Latent Diffusion Reasoning Prevents Entropy Collapse in Reinforcement Learning** | [abs](https://arxiv.org/abs/2602.01705) | Runs RL with a continuous latent-diffusion policy over reasoning trajectories decoded to text, using multiple completions per latent to avoid entropy collapse. |
| 2026-02 | **Diffusion-State Policy Optimization for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.06462) | DiSPO adds credit assignment over intermediate masked states by resampling fillings without extra rollouts, improving on diffu-GRPO and SPG on LLaDA-8B math/planning. |
| 2026-02 | **Efficient and Stable Reinforcement Learning for Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.08905) · [code](https://github.com/Lolo1222/STP) | Spatio-Temporal Pruning restricts the search space and skips late refinement steps to make dLLM RL cheaper and lower-variance. |
| 2026-03 | **Stabilizing Reinforcement Learning for Diffusion Language Models** | [abs](https://arxiv.org/abs/2603.06743) | StableDRL counters GRPO instability from noisy importance-ratio estimates in dLLMs with unconditional clipping and self-normalization. |
| 2026-03 | **Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages** | [abs](https://arxiv.org/abs/2603.12554) · [code](https://github.com/vishnutez/egspo-dllm-rl) | Exact policy gradient over denoising steps, with entropy-selected update steps and one-step-reward stepwise advantages, reporting SOTA on coding and logic benchmarks. |
| 2026-04 | **DARE: Diffusion Large Language Models Alignment and Reinforcement Executor** | [abs](https://arxiv.org/abs/2604.04215) | An open framework unifying SFT, PEFT, preference optimization, and RL for dLLMs with reproducible evaluation. |
| 2026-05 | **Relative Score Policy Optimization for Diffusion Language Models** (ICML'26) | [abs](https://arxiv.org/abs/2605.10218) | RSPO uses the reward advantage as a target for the noisy relative log-ratio and updates on the gap, stabilizing RLVR for dLLMs with strong planning gains. |
| 2026-05 | **Block-R1: Rethinking the Role of Block Size in Multi-domain Reinforcement Learning for Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2605.11726) · [code](https://github.com/YanJiangJerry/Block-R1) | Shows block size causes cross-domain conflicts in dLLM RL and releases the Block-R1-41K dataset, a benchmark, and per-sample best-block-size training. |
| 2026-05 | **DACA-GRPO: Denoising-Aware Credit Assignment for Reinforcement Learning in Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.16342) | A plug-in for GRPO-style dLLM trainers that weights tokens by denoising progress and reduces likelihood-estimate bias with stratified masking. |
| 2026-05 | **GDSD: Reinforcement Learning as Guided Denoiser Self-Distillation for Diffusion Language Models** | [abs](https://arxiv.org/abs/2605.29398) · [code](https://github.com/GaryBall/GDSD) | Recasts dLLM RL as distilling the denoiser toward an advantage-guided teacher, avoiding the biased ELBO surrogate, with up to 19.6% gains on planning, math, and coding. |
| 2026-06 | **Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.04396) | CAPR condenses denoising traces into path states and cached branches and trains a block-level value head, giving tree-like credit at ~0.6-0.75x rollout cost. |
| 2026-06 | **Back on Track: Aligning Rewards and States for Reasoning in Diffusion Large Language Models** (EMNLP'26) | [abs](https://arxiv.org/abs/2606.08501) | PAPO converts sparse terminal rewards into step-aware process rewards and replays real trajectories at high-entropy steps, improving GSM8K, MATH500, Countdown, and Sudoku. |
| 2026-06 | **Efficient Reinforcement for Visual-Textual Thinking with Discrete Diffusion Model** | [abs](https://arxiv.org/abs/2606.14792) | Shows discrete diffusion makes RL over interleaved image-text reasoning cheaper than AR and adds factorized reward assignment to reduce cross-modal interference. |
| 2026-06 | **Learning from the Self-future: On-policy Self-distillation for dLLMs** | [abs](https://arxiv.org/abs/2606.18195) · [code](https://github.com/xingzhejun/d-OPSD) | d-OPSD conditions the student on the model's own generated answers and supervises at the denoising-step level, beating RLVR and SFT on four reasoning benchmarks with ~10% of RLVR's steps. |
| 2026-06 | **SLIM-RL: Risk-Budgeted Random-Masking RL for Diffusion LLMs Without Trajectory Slicing** | [abs](https://arxiv.org/abs/2607.00208) · [code](https://github.com/laolaorkkkkk/SLIM-RL) | Bounds per-step commit risk with a budgeted decoder and trains with a random-masking objective instead of TraceRL-style trajectory slicing, matching prior SOTA with fewer training samples. |
| 2026-07 | **dOPSD: On-Policy Self-Distillation for Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.04428) | Uses later steps of a dLLM's own denoising trajectory as the teacher signal instead of ground-truth labels, improving math and code on Dream and LLaDA. |
| 2026-07 | **Mask-Aware Policy Gradients for Diffusion Language Models** (COLM'26) | [abs](https://arxiv.org/abs/2607.15200) · [code](https://github.com/Haran71/mask-aware-policy-gradients) | Models each MDLM step as a two-stage action (token fill + remask) MDP and optimizes both policy-gradient terms, reporting 87.1% GSM8K and 53.4% MBPP. |
| 2026-07 | **Trace-Based On-Policy Distillation for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.16872) | TOPD trains a masked DLM on its own denoising traces with token-level reverse-KL teacher supervision, gaining reasoning ability without reward modeling. |
| 2026-08 | **Beyond On-Policy Exploration: Integrating External Policy Rollouts for Reinforcement Learning in Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.01717) | ERILS mixes length-controlled, separately scored rollouts from a stronger external model into dLLM RL, improving multi-sample performance on Sudoku, Countdown, and MATH500. |
| 2026-09 | **CanvasAnneal: Curriculum Reinforcement Learning for Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.13060) | Seeds dLLM generation with a stronger teacher's reasoning and gradually withdraws it as an RL curriculum, speeding up RL with task-dependent gains. |
| 2026-10 | **Enhancing Diffusion Language Models with Autoregressive Post-Training Weights** | [abs](https://arxiv.org/abs/2610.08108) | A2D transfers existing AR post-training (SFT/RL) weight deltas to diffusion LMs without training, improving instruction following, math, and coding. |
| 2026-10 | **On-Policy Residual Learning for Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.20687) | ORL trains DLMs on the residual predictions produced during their own denoising with a sharpening objective, reaching up to 6.2x more tokens per forward pass on a 4B model. |
| 2026-10 | **Reinforcement Learning-driven Diffusion Models: Language, Vision, and Beyond** (TPAMI'26) | [abs](https://arxiv.org/abs/2610.21539) · [code](https://github.com/MAIR-Lab-HUST/Awesome-RL-for-Diffusion-Models) | Survey of RL post-training for diffusion models across language, multimodal, robotic, and scientific domains, with empirical comparison of RL strategies. |

### 19 · 8. Multimodal & unified diffusion

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-02 | **LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models** (ICML'26) *(unified)* | [abs](https://arxiv.org/abs/2602.14147) | A multimodal reasoning dLLM built on LaViDa-O, post-trained with unified SFT and multi-task RL (answer-forcing, tree search, complementary likelihood estimation) for visual math, grounding, and editing. |
| 2026-03 | **Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion** (ICML'26) *(unified)* | [abs](https://arxiv.org/abs/2603.06577) | Described as the first any-to-any multimodal LM built entirely on mask-based discrete diffusion, covering text, speech, and images. |
| 2026-03 | **MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Unified Multi-Modal Instruction and Generation** (ACM MM'26) *(VLA)* | [abs](https://arxiv.org/abs/2603.25406) | A fully native, pretrained discrete-diffusion VLA built on MMaDA that unifies multimodal understanding, generation, and robot action prediction. |
| 2026-03 | **Dynin-Omni: Omnimodal Unified Large Diffusion Language Model** *(unified)* | [abs](https://arxiv.org/abs/2604.00007) | A masked-diffusion omnimodal foundation model that unifies text, image, and speech understanding and generation, plus video understanding, in one architecture. |
| 2026-04 | **LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation with Diffusion Large Language Model** *(unified)* | [abs](https://arxiv.org/abs/2604.20796) · [code](https://github.com/inclusionAI/LLaDA2.0-Uni) | An MoE dLLM that discretizes images with SigLIP-VQ and runs block-level masked diffusion over text and vision tokens, with a diffusion decoder for images. |
| 2026-06 | **TBD-VLA: Temporal Block Diffusion Vision Language Action Model** *(VLA)* | [abs](https://arxiv.org/abs/2606.07895) | Applies masked discrete diffusion within temporal action blocks and AR generation across blocks, which allows real-time chunking through temporal in-painting. |
| 2026-06 | **PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models** *(multimodal)* | [abs](https://arxiv.org/abs/2606.19534) · [code](https://github.com/MSALab-PKU/PerceptionDLM) | A multimodal DLM that captions multiple image regions in parallel through structured attention masking, with a new Parallel Detailed Localized Captioning benchmark. |
| 2026-06 | **dVLA-RL: Reinforcement Learning over Denoising Trajectories for Discrete Diffusion Vision-Language-Action Models** *(VLA)* | [abs](https://arxiv.org/abs/2606.23623) | Runs PPO-style RL for discrete-diffusion VLAs on the joint probability of the sampled denoising path instead of the intractable marginal action likelihood. |
| 2026-07 | **Discrete Diffusion Language Models for Interactive Radiology Report Drafting** *(multimodal)* | [abs](https://arxiv.org/abs/2607.01436) | LoRA-adapts DiffusionGemma-26B for medical VQA, where it matches or beats same-size AR Gemma-4-26B, decodes 3.5-4.4x faster, and supports any-order infill for radiologist-edited reports. |
| 2026-07 | **Reinforcing the Generation Order of Multimodal Masked Diffusion Models** *(unified)* | [abs](https://arxiv.org/abs/2607.08056) | Shows that logits alone cannot set a good unmasking order for text-to-image generation and multimodal understanding, and learns an order-control module with GRPO. |
| 2026-07 | **TS-Mask VLA: 2D Temporal-Spatial Masking for Vision-Language-Action Model with Effective Bridging** *(VLA)* | [abs](https://arxiv.org/abs/2607.09818) | A VLA with a discrete diffusion action expert, conditioned on the VLM through bridge attention, that uses 2D temporal-spatial masking of action sequences. |
| 2026-07 | **Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes** *(unified)* | [abs](https://arxiv.org/abs/2607.13188) | SC-CMJP couples text and image masked-diffusion sampling so each modality's transition rates depend on the other's confidence, and adds a remasking jump that retracts contradictory tokens. |
| 2026-07 | **Seeing the End at Step Zero: Accelerating Diffusion MLLMs via MLP Sparsity-Aware Truncation** (ACM MM'26) *(multimodal)* | [abs](https://arxiv.org/abs/2607.14557) | Seer detects the true output length at the first denoising step from a shift in MLP activation sparsity, then truncates the padded [EOS] suffix in one shot. |
| 2026-07 | **ST-Veto: Spatio-Temporal Token Veto for Diffusion MLLMs via Taylor Prediction and Visual Grounding** (ICML'26) *(multimodal)* | [abs](https://arxiv.org/abs/2607.17884) | A training-free method that vetoes temporally unstable tokens (second-order Taylor prediction of confidence) and weakly image-grounded tokens during dMLLM unmasking, improving multimodal reasoning. |
| 2026-07 | **Faster but Different: Diagnosing and Controlling Content Drift in Accelerated Multimodal Diffusion Language Models** *(multimodal)* | [abs](https://arxiv.org/abs/2607.29079) | On 300 images with LLaDA-V, shows that training-free acceleration (Fast-dLLM) silently changes generated content, and that the KV-cache refresh interval is the main consistency control. |
| 2026-08 | **WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA** *(VLA)* | [abs](https://arxiv.org/abs/2608.01035) | Converts a pretrained AR driving VLA into a multi-task discrete diffusion VLA through block-wise adaptation, block-wise distillation, and cross-scale distillation. |
| 2026-08 | **DAVET: Denoising-Aware Visual Evidence Trajectory Allocation for Diffusion Vision-Language Models** *(multimodal)* | [abs](https://arxiv.org/abs/2608.01821) | A training-free accelerator that allocates visual tokens per denoising step according to the evolving generation state, because dVLMs need different amounts of visual evidence at different steps. |
| 2026-08 | **Does More Retrieved Evidence Help Visual Retrieval-Augmented Generation with Diffusion Language Models?** *(multimodal)* | [abs](https://arxiv.org/abs/2608.07006) · [code](https://github.com/wjkuser/ECF) | Shows that passing more retrieved pages to diffusion LMs often lowers visual-RAG accuracy through source conflict in parallel denoising, and proposes ECF, a training-free entropy-based evidence filter. |
| 2026-08 | **Context-Aware Cluster Decoding: Semantic Anchor-Driven Coherence in dMLLMs** (EMNLP'26) *(multimodal)* | [abs](https://arxiv.org/abs/2608.22367) · [code](https://github.com/zhaoyk-sysu/CACD-dMLLM) | Training-free, block-free decoding that scores masked positions by confidence × neighbor proximity to reduce semantic drift and repetition in long dMLLM outputs. |
| 2026-08 | **Visual Information-Guided Parallel Decoding for Diffusion Multimodal Large Language Models** *(multimodal)* | [abs](https://arxiv.org/abs/2608.26580) | VIG-Sampler chooses which masked tokens to decode in dMLLMs by their attention to image tokens, instead of by frequency-biased certainty. |
| 2026-08 | **Information-Guided Frontier Decoding: Contextual Utility-Driven Commitment in dMLLMs** (EMNLP'26) *(multimodal)* | [abs](https://arxiv.org/abs/2608.26641) | IGFD is a training-free unmasking order for dMLLMs that ranks tokens by confidence, neighborhood uncertainty, and structural risk, so semantic anchors are committed before fragile tokens such as punctuation. |
| 2026-09 | **Reliability Challenges in Diffusion Vision-Language Models** (EMNLP'26) *(multimodal)* | [abs](https://arxiv.org/abs/2609.01318) | First systematic hallucination and bias evaluation of six diffusion LVLMs against AR baselines; finds a reversed yes-bias, degraded linguistic quality, and demographic and multiple-choice-length failure modes. |
| 2026-09 | **DIVA: Exploiting Cross-Step Conditional Propagation for Visual Jailbreaks in Discrete Diffusion Vision-Language Models** (EMNLP'26) *(multimodal)* | [abs](https://arxiv.org/abs/2609.05525) · [code](https://github.com/loststars2002/DIVA) | Identifies a diffusion-specific jailbreak vulnerability: adversarial visual semantics are re-applied and amplified at every denoising step. Proposes DIVA, a white-box visual attack on dVLMs. |
| 2026-09 | **Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model** *(VLA)* | [abs](https://arxiv.org/abs/2609.13053) | Builds on the Dynin-Omni masked-diffusion backbone and treats language, observations, goals, and actions as discrete tokens, so one model learns action, next-observation, goal, and instruction prediction. |
| 2026-09 | **LLaDA-UI: Bringing Block-wise Diffusion to Vision-Language GUI Agents** *(multimodal)* | [abs](https://arxiv.org/abs/2609.13287) | A 16.7B MoE block-diffusion GUI agent on the LLaDA2.0-mini backbone that beats Qwen2.5-VL-7B and surpasses Qwen3-VL-8B on 4 of 6 GUI grounding/navigation benchmarks. |
| 2026-09 | **Before the Token Commits: Trajectory-Level Benchmarking of Visual Hallucinations in Diffusion VLMs** *(multimodal)* | [abs](https://arxiv.org/abs/2609.34772) | Introduces DynaHall, a benchmark that records the answer tendency at every unmasking step; across five diffusion VLMs, hallucinations are often settled before any answer token is revealed. |
| 2026-09 | **Humanoid Loco-Manipulation With Discrete VLA Model** *(VLA)* | [abs](https://arxiv.org/abs/2609.35709) | Holo-M extends an LM vocabulary with four body-part action tokenizers for humanoid whole-body control and decodes them with grouped discrete diffusion; it scores highest on the SIMPLE benchmark. |
| 2026-09 | **Gestalt: Large Multimodal Interplay Model** *(unified)* | [abs](https://arxiv.org/abs/2610.00576) | A unified discrete-diffusion multimodal model whose interplay-partitioned architecture uses learnable interplay tokens for cross-modal exchange; evaluated on image generation, multimodal understanding, and text. |
| 2026-10 | **Two Clocks in Diffusion MLLMs: When Answers Stabilize Before Rationales Unfold** (NeurIPS'26 Workshop) *(multimodal)* | [abs](https://arxiv.org/abs/2610.00953) | Shows that in masked-diffusion MLLMs such as LaViDa, the answer candidate often stabilizes while 89-98% of the rationale canvas is still unwritten, and that block length controls this effect. |
| 2026-10 | **SUAVE: Unified Video-Action Models via Masked Diffusion** *(VLA)* | [abs](https://arxiv.org/abs/2610.04009) | One masked diffusion transformer generates video and actions from language, with all three modalities as discrete tokens in one vocabulary; choosing what to mask changes the task. |
| 2026-10 | **Omni-Diffusion-Distill: Few-Step Distillation of Unified Multimodal Diffusion Large Language Models** *(unified)* | [abs](https://arxiv.org/abs/2610.10990) | A two-stage distillation in discrete token space that compresses a unified multimodal dLLM into a few-step student while keeping both image generation and understanding. |
| 2026-10 | **Intern Lumina U2: A Multi-Codebook Diffusion Large Language Model for Omni-Visual Understanding and Image Generation** *(unified)* | [abs](https://arxiv.org/abs/2610.20818) | Described as the first diffusion LM to represent multi-codebook visual tokens in a fully discrete space shared with text, for unified visual understanding and image generation. |
| 2026-10 | **CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective** *(multimodal)* | [abs](https://arxiv.org/abs/2610.23290) | Traces the token repetition caused by approximate KV caching in diffusion MLLMs to disrupted context-token information flow, and proposes a training-free plug-in fix. |
| 2026-10 | **Beyond the Sequentiality Bottleneck: A Hybrid Diffusion-Autoregressive Framework for Efficient Multimodal Reasoning** *(multimodal)* | [abs](https://arxiv.org/abs/2610.28584) | CoRAL uses a DLM planner to generate K reasoning trajectories in parallel (trained with reward-guided denoising), and a large AR executor then picks a plan and writes the answer on vision-language benchmarks. |

### 19 · 9. Inference & acceleration

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-10 | **Attention Is All You Need for KV Cache in Diffusion LLMs** | [abs](https://arxiv.org/abs/2510.14973) · [code](https://github.com/VILA-Lab/Elastic-Cache) | Elastic-Cache adaptively refreshes dLLM KV caches (block-wise caching of distant MASK tokens, depth-dependent selective refresh) to cut redundant QKV recomputation. |
| 2026-02 | **DFlash: Block Diffusion for Flash Speculative Decoding** (ICML'26) | [abs](https://arxiv.org/abs/2602.06036) · [code](https://github.com/z-lab/dflash) | Uses a lightweight block-diffusion model as a parallel drafter for speculative decoding of autoregressive LLMs. |
| 2026-06 | **WaveFilter: Enhancing the Long-Context Capability of Diffusion LLMs via Wavelet-Guided KV Cache Filtering** | [abs](https://arxiv.org/abs/2606.00724) | Uses wavelet-guided filtering to select critical tokens for KV caching, so dLLMs keep generation quality on ultra-long contexts while cutting cost. |
| 2026-06 | **SimSD: Simple Speculative Decoding in Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.02544) · [code](https://github.com/airevo2/SimSD-release) | A training-free method that enables token-level speculative decoding in dLLMs through a temporal causal attention mask and RoPE alignment. |
| 2026-06 | **SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding** | [abs](https://arxiv.org/abs/2606.04974) · [code](https://github.com/TH-AI-Lab-PKU/SAID) | Spends denoising steps on "scaffold" tokens that fix coarse semantic structure, then fills predictable detail tokens in fewer steps; also adapted to block-wise dLLMs. |
| 2026-06 | **Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models** (EMNLP'26) | [abs](https://arxiv.org/abs/2606.10537) · [code](https://github.com/menik1126/Prefilling-dLLM) | A training-free prefill-decode disaggregation that caches chunked prefix KV once and attends only to the top-K relevant chunks, making per-step cost quadratic only in the decode length. |
| 2026-06 | **Walk fast but be careful: Understanding Parallel Sampling in Masked Diffusion** | [abs](https://arxiv.org/abs/2606.22976) | Shows that unmasking by scores such as lowest entropy is not uniformly better than random parallel sampling, because accuracy depends on the conditional dependence structure, and proposes training-free bisection samplers. |
| 2026-07 | **TACG: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding** | [abs](https://arxiv.org/abs/2607.03236) · [code](https://github.com/Clarence-CV/TACG-DLLM) | A training-free decoder that keeps token identities from the base posterior but uses cross-step trajectory signals to decide when a token is ready to commit. |
| 2026-07 | **Sangam: Efficiently Serving Diffusion LLMs with the AR Stack** | [abs](https://arxiv.org/abs/2607.04206) · [code](https://github.com/UT-InfraAI/sangam) | A serving system for cached dLLM inference that adapts AR serving mechanisms (recurring prefills, block-sized decodes) with a central cluster router and worker-local iteration-level schedulers. |
| 2026-07 | **BlockServe: Block-Grained Continuous Batching for High-Throughput Diffusion LLM Serving** | [abs](https://arxiv.org/abs/2607.08930) | Continuous batching that evicts finished requests at block boundaries and extends dual cache and parallel decoding to mixed-state batches, reporting 1.9-10.6x throughput over Fast-dLLM on Dream and LLaDA. |
| 2026-07 | **Polestar: Drift-Aware Cache Calibration and Token Commitment for Efficient Inference of Diffusion LLMs** | [abs](https://arxiv.org/abs/2607.14107) | A training-free framework that uses token-representation drift across steps as one signal to both refresh stale KV cache and set token-commitment parallelism. |
| 2026-07 | **LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2607.16339) | Training-free acceleration for semi-autoregressive dLLM decoding that losslessly memoizes block-invariant intermediate results (embeddings, RoPE, etc.) and adds mixed precision. |
| 2026-07 | **FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.17652) | Uses token-to-token editing in self-correcting block dLLMs to decode downstream blocks from drafts in a gated wavefront with a readiness gate, without retraining. |
| 2026-07 | **Multi-Mask Diffusion Language Models for Few-Step Generation** (COLM'26) | [abs](https://arxiv.org/abs/2607.19686) | MultiMDM uses multiple mask states so forward trajectories keep terminal entropy, which enables consistency-style few-step generation while preserving the masking structure. |
| 2026-07 | **PRESTO: Prefix-Aligned Tree Drafting for Diffusion Speculative Decoding** | [abs](https://arxiv.org/abs/2607.22634) | Replaces linear drafting with prefix-aligned tree drafting for dLLM drafters in speculative decoding, using the multiple candidate tokens per position to increase acceptance length. |
| 2026-07 | **Commit Locally, Exit Globally: Coordinating Adaptive Sampling and Early Exit in Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.28166) · [code](https://github.com/ming053l/C4-dLLM) | C^4 is a training-free framework that combines local confidence-based commitment with an adaptive global early-exit gate based on answer confidence and temporal agreement. |
| 2026-08 | **OPTD: On-Policy Transition Distillation with Consistency-Guided Adaptive Compression for Few-Step Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.02942) | Few-step distillation on the student's own on-policy trajectories, with a consistency signal that adaptively sets how many teacher steps each student transition compresses. |
| 2026-08 | **Archer: Adaptive Reuse of Cached Hidden States for Efficient Rollback in Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.08086) · [code](https://github.com/Hxnng/Archer) | A training-free caching method for rollback-capable DLMs that recomputes the mutable response each update while reusing prompt K/V within a bounded neighborhood. |
| 2026-08 | **CForce: Boosting Parallel Decoding for dLLMs via Consistency Forcing** | [abs](https://arxiv.org/abs/2608.13925) · [code](https://github.com/inclusionAI/dFactory) | A distillation method trained on self-rollout trajectories that aligns early-stage mask predictions with later-stage ones, with a confidence-adaptive KL loss, so aggressive parallel decoding stays accurate. |
| 2026-08 | **Affix Cache for Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2608.26140) | ACache reuses KV states across requests for shared spans at any position (prefix, infix, or suffix) and recomputes only a small, influence-selected subset of affix tokens. |
| 2026-08 | **Trajectory-Level Speculative Decoding for Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.27514) | Drafts multi-token denoising trajectories by confidence-stratified tree exploration and verifies them blockwise in parallel, adding inter-block speculation via bidirectional lookahead. |
| 2026-09 | **GroupKV: Hierarchical KV Cache Management for Long-Context Diffusion LLM Inference** | [abs](https://arxiv.org/abs/2609.17573) | Group-level coarse-to-fine sparse KV selection with cross-layer predictive prefetching and staleness correction, extending servable context up to 48x under constrained GPU memory. |
| 2026-09 | **Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs** | [abs](https://arxiv.org/abs/2609.26796) · [code](https://github.com/VILA-Lab/Flash-dLLM) | A training-free framework that co-designs KV caching and parallel decoding around the I/O bottlenecks that appear when cache reuse and parallel verification run together. |
| 2026-09 | **Reliable Parallel Decoding in Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.36452) | Shows that confidence alone does not give a reliable commitment order and proposes RPD, a training-free method that commits candidates whose predictions are stable across the final layers. |
| 2026-10 | **Mask-Guided KV Cache Eviction in Block Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.06996) | MaskAhead is a training-free method that ranks KV entries with mask queries: current-block masks guide selection and probes of upcoming masked blocks guide eviction. |
| 2026-10 | **PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.17375) · [code](https://github.com/Quentin-Fruytier/PIG-DLM-release) | Trains an auxiliary model that estimates, in one forward pass, how far parallel decoding diverges from sequential decoding, and uses it to decide which tokens to commit together. |
| 2026-10 | **On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction** | [abs](https://arxiv.org/abs/2610.22543) | Observes that the previous pass already predicts the right token at 92-100% of positions, and makes speculative decoding work for DLMs by predicting which position commits next. |
| 2026-10 | **Balance-Aware Pivot Selection for Parallel Diffusion LLM Decoding** | [abs](https://arxiv.org/abs/2610.26040) | BPD picks central pivot positions inside masked intervals instead of relying only on confidence, which shapes the remaining decoding schedule for higher parallelism. |
| 2026-10 | **TeCoP: Accelerating Diffusion Large Language Model Decoding via Tentative Context Pipelining** | [abs](https://arxiv.org/abs/2610.27583) | A training-free decoder that pipelines blocks and reuses stable but uncommitted predictions as provisional context, reporting up to 7.50x throughput without KV cache (11.77x with) on LLaDA and Dream. |

### 19 · 10. Guidance, control & constrained decoding

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-05 | **DINGO: Constrained Inference for Diffusion LLMs** | [abs](https://arxiv.org/abs/2505.23061) | A dynamic-programming constrained decoding strategy that enforces regular-expression constraints on diffusion LLM outputs while provably preserving the output distribution. |
| 2026-06 | **EPIC: Efficient and Parallel Inference under CFG Constraints for Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.00722) · [code](https://github.com/hyundong98/EPIC-Decoding) | Speeds up context-free-grammar-constrained dLLM decoding with lexing memoization, Earley-style parsing, and parallel commit of compatible tokens, reporting up to 67.5% less inference time than prior CFG methods. |
| 2026-06 | **VGB for Masked Diffusion Model: Efficient Test-time Scaling for Reward Satisfaction and Sample Editing** | [abs](https://arxiv.org/abs/2606.28301) | MDM-VGB extends the Jerrum-Sinclair backtracking random walk to a masked-state graph, giving reward-guided remasking at arbitrary positions for test-time scaling. |
| 2026-07 | **Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata** | [abs](https://arxiv.org/abs/2607.07026) | An exact, tractable algorithm to sample from the constrained mean-field posterior at each denoising step for any finite-automaton constraint (e.g., JSON schemas), plus a log-depth sampling scheme. |
| 2026-08 | **Commitment Before Realization: When Classifier-Free Guidance Becomes Unnecessary in Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2608.08082) | Measures the remaining value of CFG along MDLM decoding, finds it is prompt-specific and concentrated early, and uses this to decide when guidance can be switched off. |
| 2026-08 | **Discrete Diffusion Inference-Time Control with Nested Sequential Monte Carlo** | [abs](https://arxiv.org/abs/2608.20123) | Formulates nested SMC and fully-adapted nested SMC for Feynman-Kac steering of discrete diffusion LMs toward sequence-level rewards, correcting biased prior formulations. |
| 2026-09 | **Constraints Are Graphs, Not Chains: Exact Decoding for Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.32900) | FactorDLM is a training-free exact constrained decoder that encodes relational constraints as factor graphs, which stay O(k) where automata or CFG encodings grow exponentially. |
| 2026-09 | **Twist, Don't Tilt: Trajectory-Exact Constrained Decoding for Masked Diffusion Models** | [abs](https://arxiv.org/abs/2609.35609) | Proves that composing per-step exact automaton-constrained draws biases MDLM outputs away from the model's distribution over valid trajectories, derives the exact bias, and corrects it to obtain trajectory-exact constrained decoding. |
| 2026-09 | **FastGuide: Accelerating Reward Guidance for Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2609.36202) | Speeds up gradient-based reward guidance for masked dLLMs by computing guidance once per step and reusing it for multiple tokens, adaptively mixing parallel and autoregressive decoding. |

### 19 · 11. Applications

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-01 | **Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars** | [abs](https://arxiv.org/abs/2602.00612) | LAVE uses per-position token distributions to look ahead and verify that each proposed token can still complete to a CFG-valid output. |
| 2026-06 | **Self-Augmenting Retrieval for Diffusion Language Models** (ICML'26) | [abs](https://arxiv.org/abs/2606.06474) | SARDI uses low-confidence tokens discarded during denoising as lookahead retrieval queries, a training-free dynamic RAG method that wins on five multi-hop QA sets at up to 8x throughput. |
| 2026-06 | **TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering** | [abs](https://arxiv.org/abs/2606.08408) | Extends activation steering to dLLMs to edit existing text in place toward a target concept while preserving structure (IMDB sentiment, Cats-and-Dogs). |
| 2026-06 | **Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of Evolving Contexts** | [abs](https://arxiv.org/abs/2606.12807) | Locates outdated summary spans, remasks them, and rewrites them with a diffusion LM, with a new StreamSum benchmark; the method is faster and more controllable than full regeneration. |
| 2026-08 | **Length-Adaptive Decoding for Masked Diffusion Machine Translation** (EMNLP'26) | [abs](https://arxiv.org/abs/2608.22274) · [code](https://github.com/Entropy-Valley/Entropy-Valley) | Entropy-Valley picks the target length for masked diffusion MT from the average predictive entropy of all-mask passes, improving COMET-22 on En-Zh, Zh-En, and En-De. |
| 2026-09 | **Predict, Don't Iterate: Efficient Adaptive-Length Infilling for Diffusion Language Models** (EMNLP'26) | [abs](https://arxiv.org/abs/2609.02108) · [code](https://github.com/Hsu1023/PILL) | PILL predicts the infill span length instead of iterating, beating the strongest baseline across five models and eight benchmarks at about 1.82x speed. |
| 2026-09 | **In-Place Instruction Following in Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.07160) | Defines position-anchored instruction following for dLLMs, releases IIF-Bench, and proposes GRAFT post-training (about +15 IIF points on four models). |
| 2026-10 | **Multi-Target Few-Shot Prompting with Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.22294) | Introduces few-shot prompting in which demonstrations guide the filling of several masked regions in one shared text, using selective masking and a likelihood-ratio reward. |

### 19 · 12. Safety, watermarking & robustness

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2025-09 | **A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models** (ICLR'26) | [abs](https://arxiv.org/abs/2509.23286) | Token-level alignment that trains dLLMs to emit an EOS refusal wherever harmful content appears, cutting DIJA success from >80% to near zero (project page ai-isl.github.io/A2D). |
| 2025-09 | **TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2510.01274) | Treats denoising steps as an action trace and selects the most informative sub-trace for hallucination detection, with +15.2% average AUROC. |
| 2025-10 | **Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerability** (ICLR'26) | [abs](https://arxiv.org/abs/2510.00565) · [code](https://github.com/mdl-lab/dlm-priming-vulnerability) | Shows that affirmative tokens injected mid-denoising jailbreak dLLMs, and proposes alignment from contaminated intermediate states. |
| 2025-11 | **Watermarking Discrete Diffusion Language Models** | [abs](https://arxiv.org/abs/2511.02083) | Applies Gumbel-max sampling at each diffusion step, seeded by sequence position, for distortion-free watermarks that are reliably detected on LLaDA. |
| 2026-01 | **dgMARK: Decoding-Guided Watermarking for Diffusion Language Models** (ICML'26) | [abs](https://arxiv.org/abs/2601.22985) | Embeds the watermark in the unmasking order via a hash-parity rule without changing token probabilities, with a sliding-window detector robust to edits. |
| 2026-03 | **Characterizing Memorization in Diffusion Language Models: Generalized Extraction and Sampling Effects** | [abs](https://arxiv.org/abs/2603.02333) | A probabilistic extraction framework for DLMs shows that higher sampling resolution increases exact extraction, and that DLMs leak less PII than AR models under aligned prefix evaluation. |
| 2026-05 | **BadDLM: Backdooring Diffusion Language Models with Diverse Targets** | [abs](https://arxiv.org/abs/2605.09397) | A backdoor framework for DLMs with targets ranging from concept injection to alignment bypass, while preserving clean performance. |
| 2026-05 | **Uncertainty Quantification for Large Language Diffusion Models** | [abs](https://arxiv.org/abs/2605.14570) | First in-depth UQ study for dLLMs, using training-free denoising-trajectory signals and masked-diffusion likelihoods, near the best sampling baselines at up to 100x lower cost. |
| 2026-06 | **TRE: Training-Free Hallucination Detection for Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.22661) | Detects dLLM hallucinations from a single generation by combining token-level uncertainty with late-step entropy. |
| 2026-07 | **Beyond the Bidirectional Promise: Re-evaluating the Robustness of Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.27386) | Finds that LLaDA-8B and Dream-7B resist gradient-based adversarial suffixes but not natural noise, are overconfident, and fail mainly in decoding rather than in encoding. |
| 2026-08 | **Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits** | [abs](https://arxiv.org/abs/2608.07430) · [code](https://github.com/ellyoana/sn-guided-diffusion) | Finds that dLLM safety is carried by sparse safety neurons inherited from AR models, and introduces the SN-Guided Diffusion black-box jailbreak (up to 77.1% transfer ASR on Llama-3-8B-Instruct). |
| 2026-08 | **Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models** (EMNLP'26) | [abs](https://arxiv.org/abs/2609.00495) · [code](https://github.com/Glresearch1/RAEC) | RAEC is a training-free decoder that commits to persistent early-step refusal signals, lowering ASR on LLaDA and Dream. |
| 2026-09 | **Membership Inference in Fine-tuned Diffusion Language Models via Token-level Memorization Asymmetry** (EMNLP'26) | [abs](https://arxiv.org/abs/2609.00873) | Identifies token-level memorization asymmetry and proposes the Q-Skew membership-inference indicator for fine-tuned DLMs, which also enables PII extraction. |
| 2026-09 | **DenMark: Robust Semantic Watermarking for Diffusion Language Models** | [abs](https://arxiv.org/abs/2609.14257) · [code](https://github.com/TianhaoMa5/DenoMark) | A semantic watermark that uses temporary rollouts during denoising to pick high-score updates, with a calibrated detector robust to paraphrase. |
| 2026-09 | **Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis** (NeurIPS'26) | [abs](https://arxiv.org/abs/2609.30841) | Models dLLM alignment as an energy barrier and proposes three training-free jailbreak-detection signals (initial-token probability and two trajectory-dynamics signals). |
| 2026-10 | **Noise Out, Bias In: Targeted Bias Injection in Diffusion Language Models via Closed-Loop Activation Steering** | [abs](https://arxiv.org/abs/2610.05894) | Feedback-controlled activation steering during denoising pushes a frozen LLaDA-8B-Instruct toward adversary-chosen demographic answers on BBQ and SocialStigmaQA. |
| 2026-10 | **The Tokens Strike Back: Safety-Aware Denoising with Directional Embedding-based Revision** | [abs](https://arxiv.org/abs/2610.17636) | SADDER extends the training-free Safety-Aware Denoiser with an embedding-space safety signal that selectively revises unsafe tokens while keeping the denoiser exact. |

### 19 · 13. Efficiency: quantization

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-06 | **STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models** | [abs](https://arxiv.org/abs/2606.04945) | PTQ for dLLMs that targets activation gaps between masked and unmasked tokens and error buildup across denoising steps, using a state-guided transformation plus temporal correction. |
| 2026-06 | **FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models** (ICML'26) | [abs](https://arxiv.org/abs/2606.06547) | Two-stage PTQ calibration that reweights layer-wise reconstruction toward fragile positions at the decoding "write frontier," where quantization error flips irreversible commitments. |
| 2026-09 | **Hardware Acceleration of Block-Diffusion LLM for Edge Devices** | [abs](https://arxiv.org/abs/2609.01084) | Hardware-software co-design for batch-one edge inference of block-diffusion LLMs: wide-I/O LPDDR, low-rank plus INT8-residual prefix KV, and low-bit FFN delta schemes. |
| 2026-10 | **ITC-MoE: Importance-guided Token-aware Compression for MoE Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.01296) · [code](https://github.com/lianjunl13-sudo/ITC-MoE) | Low-rank compression for MoE DLMs that allocates rank by mode-specific redundancy and token-wise (hot vs. cold) expert utilization, not static factorization. |
| 2026-10 | **SoloQ: Calibration-Free Quantization for Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.07121) | Low-bit weight-activation quantization for dLLMs that needs no calibration data, motivated by activation shifts across masking states and denoising steps. |

### 19 · 14. Embeddings, energy-based & Bayesian flow networks

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-02 | **Embedding Inversion via Conditional Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.11047) · [code](https://github.com/jina-ai/embedding-inversion-demo) | Recovers text from embeddings via conditional masked diffusion (AdaLN conditioning) in about 8 parallel denoising passes without querying the encoder. |
| 2026-02 | **Diffusion-Pretrained Dense and Contextual Embeddings** | [abs](https://arxiv.org/abs/2602.11151) | pplx-embed: multilingual retrieval embeddings trained contrastively on a diffusion-pretrained bidirectional LM, with standard and contextual (late-chunking) variants. |
| 2026-05 | **Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion** (ACL'26) | [abs](https://arxiv.org/abs/2605.04291) | Glauber-dynamics discrete diffusion with an energy built from pretrained causal or masked LMs (UL2), comparable to AR models, with strong Sudoku and Zebra results. |
| 2026-06 | **Unified Energy for Invariant and Independent Decoding in Diffusion Language Models** | [abs](https://arxiv.org/abs/2606.09159) | Uni-E combines invariant and independent energies, is computable exactly without partition estimation, and corrects the distribution shift of parallel decoding. |
| 2026-07 | **Subliminal Clocks: Latent Time Modelling in Diffusion Language Models** (EMNLP'26) | [abs](https://arxiv.org/abs/2607.01774) | Probes reveal a latent denoising-timestep signal in dLLM residual streams, and steering it predictably shifts confidence and entropy. |
| 2026-07 | **Induction in Both Directions: A Mechanistic Analysis of In-Context Learning in Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.15893) | Finds a bidirectional induction circuit in MDLMs that copies whether the source precedes or follows the target, and shows the models infer the mask ratio as an implicit timestep. |

### 19 · 15. "Language of X": biology & speech

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-03 | **LLaDA-TTS: Unifying Speech Synthesis and Zero-Shot Editing via Masked Diffusion Modeling** | [abs](https://arxiv.org/abs/2603.26364) | Replaces the AR LLM in LLM-based TTS with a masked diffusion model (fine-tuned on 50h), matching CosyVoice 3 quality with 2x LLM-stage speedup and supporting zero-shot word-level editing. |
| 2026-04 | **OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models** | [abs](https://arxiv.org/abs/2604.00688) · [code](https://github.com/k2-fsa/OmniVoice) | A diffusion-LM-style discrete NAR TTS model that maps text directly to multi-codebook acoustic tokens and covers 600+ languages zero-shot. |
| 2026-04 | **Towards A Generative Protein Evolution Machine with DPLM-Evo** (ICML'26) | [abs](https://arxiv.org/abs/2605.00182) | Extends diffusion protein LMs beyond pure masking toward edit-based, evolution-like generation, because proteins evolve through accumulated edits rather than unmasking. |
| 2026-06 | **DLLM-TTS: Block Discrete Diffusion Language Model for Text-to-Speech Synthesis** | [abs](https://arxiv.org/abs/2608.00011) | Block discrete diffusion over X-Codec2 tokens: masked diffusion within each block, blocks generated sequentially; a 0.6B model reaches RTF 0.15. |
| 2026-07 | **On the Design Space of Discrete Diffusion Online Adaptation for Molecular Optimization** | [abs](https://arxiv.org/abs/2607.02834) | Controlled study of acquisition, reward shaping, debiasing, and replay for oracle-budgeted online adaptation of discrete diffusion models, on 6 small-molecule and 3 protein-fitness tasks. |
| 2026-07 | **DELTA-TTS: Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech** (NeurIPS'26) | [abs](https://arxiv.org/abs/2607.04140) | LoRA-converts a pretrained AR TTS model into a dLLM with confidence-ordered decoding, reaching 1.75% WER on Seed-TTS test-en and 3.3x faster generation. |
| 2026-07 | **Audio-Native Speech Recognition with a Frozen Discrete-Diffusion Language Model** | [abs](https://arxiv.org/abs/2607.13013) | Adds a Whisper encoder, projector, and LoRA (42M trained parameters) to frozen DiffusionGemma-26B, a uniform-noise diffusion LM; a CTC loss grounds the audio, giving 6.6% WER on LibriSpeech test-clean in about 8 steps. |
| 2026-08 | **Multi Codec Discrete Diffusion Model for Text Guided Speech Inpainting and Editing** | [abs](https://arxiv.org/abs/2608.06424) · [code](https://github.com/iftachShoham/SIEDD) | SIEDD performs text-guided speech inpainting and editing with discrete diffusion over hierarchical codec tokens, using bidirectional acoustic context. |
| 2026-08 | **Luna-TTS Family Technical Report** | [abs](https://arxiv.org/abs/2608.11593) | Diffusion-LM TTS adapted from an AR text LLM and pretrained on 1M hours in 4 languages; it has a fully NAR RVQ-grid variant and a block-causal realtime variant (RTF 0.024). |
| 2026-09 | **One Sequence, Many Decodings: CAGenMol-2 Recasts Drug Design as Masked Molecular Inference** | [abs](https://arxiv.org/abs/2609.34301) | A masked-diffusion molecular LM that packs molecules, scalar properties, and 3D pockets into one sequence, so the masking pattern selects the task; its AdaFO optimizer raises CrossDocked2020 success from 30.2% to 70.8%. |
| 2026-09 | **RAWD-TTS: Ratio-Free Reward Alignment for Discrete-Diffusion Voice Cloning** | [abs](https://arxiv.org/abs/2609.37028) | Aligns discrete-diffusion voice cloning with ASR and speaker rewards by weighting masked-token reconstruction with group-relative advantages, with no trajectory likelihoods needed. |
| 2026-10 | **Refinement Buys Intelligibility, Search Buys Identity: What Test-Time Compute Buys in Masked-Diffusion TTS** | [abs](https://arxiv.org/abs/2610.03320) | Across 15 masked-diffusion codec TTS models, more refinement steps close 86% of the intelligibility gap but only 46% of the speaker-identity gap. |
| 2026-10 | **Beyond Token Revision: Investigating Mask-and-Replace Diffusion for Zero-Shot Text-to-Speech** | [abs](https://arxiv.org/abs/2610.09448) | Uses DeMaR (mask-and-replace training with mask-only sampling) to show that discrete-diffusion TTS gains do not require token revision at inference time. |
| 2026-10 | **DiffuPlex: Accelerating Full-Duplex Spoken Dialog Models via Rolling Masked Diffusion** | [abs](https://arxiv.org/abs/2610.12214) | Rolling masked diffusion predicts several future user and assistant frames per backbone call and revises only the unplayed future when the conversation diverges. |
| 2026-10 | **Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning** (NeurIPS'26) | [abs](https://arxiv.org/abs/2610.13330) · [code](https://github.com/bilemond/ChemDiffAgent) | ChemDiffAgent treats chemical tool-use reasoning as iterative masked denoising over interleaved reasoning and tool-call trajectories. |
| 2026-10 | **Full-duplex Spoken Language Modeling with Discrete Diffusion** | [abs](https://arxiv.org/abs/2610.17780) | Diffuplex is a full-duplex spoken LM that separates when to speak (AR interaction policy) from what to say (discrete diffusion), reaching about 13 tokens per forward pass in thinking mode. |
| 2026-10 | **Revise-TTS: A Non-Masking Refinement Framework for Masked Diffusion Models in Text-to-Speech** | [abs](https://arxiv.org/abs/2610.19499) | Lets masked-diffusion TTS revise tokens it already committed, through a non-masking refinement scheme for training and inference. |

### 19 · 16. Evaluation & benchmarks

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-02 | **Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models** | [abs](https://arxiv.org/abs/2602.19619) · [project](https://luhantang.github.io/dllm_sampler) | Uses an exact-posterior Markov-chain oracle to isolate sampler error, showing that few-step samplers stay distributionally wrong even when NLL, GenPPL, or MAUVE look good. |
| 2026-10 | **Rate-Resolved Text Evaluation with Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.15667) | Turns the MDLM masking rate into a diagnostic axis, producing rate-specific scores that add signal on SummEval and localize errors in WMT MQM zh-en. |
| 2026-10 | **Energy Consumption Analysis of Discrete Diffusion and Autoregressive Language Models** | [abs](https://arxiv.org/abs/2610.19855) | Measures inference energy of LLaDA-8B and Dream-7B against Llama-3.1-8B and Qwen-2.5-7B; at long sequences, dLLMs use about half the energy per token of KV-cached AR models. |

### 19 · 17. Masking, noise schedules & unmasking order

| Year | Paper | Links | TL;DR |
|---|---|---|---|
| 2026-06 | **Adaptive Order Policies for Masked Diffusion** | [abs](https://arxiv.org/abs/2606.00295) | Learns the unmasking order with a lightweight policy network on a masked diffusion model via a policy-reweighted MDM loss, trained either with a frozen denoiser or jointly with it. |
| 2026-07 | **Rethinking the Generation Order of Block Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.24306) | Shows block diffusion LMs are naturally aligned with left-to-right decoding and proposes PARD, a training-free sampler that keeps left-to-right unmasking while committing tokens in parallel. |
| 2026-07 | **CaRE Compute-aware Remasking Evaluation Protocol for Masked Diffusion Language Models** | [abs](https://arxiv.org/abs/2607.24763) | A compute-aware evaluation protocol that audits MDLM remasking strategies under jointly controlled step counts, metrics, and temperatures, since seven recent remasking papers used incompatible settings. |
| 2026-08 | **The data geometry of masking diffusion: Certified-optimal schedules via unmasking growth complexity** | [abs](https://arxiv.org/abs/2608.13520) | Introduces unmasking growth complexity (UGC), which bounds KL discretization error and yields optimized single- and multi-block unmasking schedules estimable from samples. |
| 2026-09 | **Schedule optimization for tau-leaping in masked discrete diffusion** | [abs](https://arxiv.org/abs/2609.21960) | Studies how to choose the denoising schedule that minimizes factorization error of parallel (tau-leaping) unmasking under a fixed sampling budget. |
| 2026-10 | **CoDR: Training-Free Confidence-Drift Remasking for Diffusion Language Models** | [abs](https://arxiv.org/abs/2610.08833) · [code](https://github.com/YueWu0301/CoDR) | Remasks committed tokens whose confidence drops between their sparse commit-time context and the denser later context, so early mistakes stop propagating. |
| 2026-10 | **LED: Improving Masked Diffusion Language Models by Lookahead Editing** | [abs](https://arxiv.org/abs/2610.23122) | Argues MDMs are easier to train but less expressive than uniform diffusion models, and augments masked diffusion decoding with a rewritable proposal state so MDMs gain iterative revision. |

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
| LLaDA 2.0 / 2.1 / 2.2 · LLaDA-MoE | [inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X) | [LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash) · [LLaDA2.0-flash](https://huggingface.co/inclusionAI/LLaDA2.0-flash) · [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) |
| iLLaDA | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | [iLLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct) |
| DiffusionGemma | — | [diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |
| Nemotron-Labs-Diffusion | — | [nemotron-labs-diffusion](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion) |
| RND1 (Radical Numerics) | [RadicalNumerics/RND1](https://github.com/RadicalNumerics/RND1) | — |
| SDAR | [JetAstra/SDAR](https://github.com/JetAstra/SDAR) | — |
| Fast-dLLM v2 (7B) | — | [Fast_dLLM_v2_7B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B) |
| d3LLM | [hao-ai-lab/d3LLM](https://github.com/hao-ai-lab/d3LLM) | — |
| Dream-Coder | [DreamLM/Dream-Coder](https://github.com/DreamLM/Dream-Coder) | — |
| DreamOn | [DreamLM/DreamOn](https://github.com/DreamLM/DreamOn) | — |
| d1 (diffu-GRPO) | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | — |
| Duo | [s-sahoo/duo](https://github.com/s-sahoo/duo) | — |
| Lumina-DiMOO | [Alpha-VLLM/Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO) | — |

### Libraries & frameworks

- **[dllm](https://github.com/ZHZisZZ/dllm)** — unified training/inference/eval toolkit (Trainer, LoRA, DeepSpeed, FSDP; recipes for LLaDA, Dream, MDLM, BD3-LM).
- **[dLLM-RL / TraceRL](https://github.com/Gen-Verse/dLLM-RL)** — post-training / RL framework for diffusion LLMs.
- **[Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM)** — fully-open diffusion LM stack (pretraining, eval, inference, checkpoints).
- **[discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance)** — classifier(-free) guidance for discrete diffusion.
- **[e2d2](https://github.com/kuleshov-group/e2d2)** — encoder-decoder diffusion LMs (NeurIPS'25).
- **[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)** — foundational score-based SDE reference impl.
- **Related awesome-lists:** [Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) · [awesome-discrete-diffusion-models](https://github.com/kuleshov-group/awesome-discrete-diffusion-models) · [RUCAIBox/Awesome-Text-Diffusion-Models](https://github.com/RUCAIBox/Awesome-Text-Diffusion-Models)
- **[dInfer](https://github.com/inclusionAI/dInfer)** — Ant Group's modular dLLM inference framework (iteration manager, decoding strategies, KV-cache manager) supporting LLaDA, LLaDA-MoE and LLaDA2.0.
- **[SGLang — Diffusion Language Models](https://docs.sglang.io/supported_models/text_generation/diffusion_language_models.html)** — Official SGLang docs for serving LLaDA2.0, SDAR and DiffusionGemma via `--dllm-algorithm` (LowConfidence, JointThreshold).
- **[llama.cpp diffusion examples](https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion)** — `llama-diffusion-cli` for running GGUF Dream, LLaDA and RND1 dLLMs locally on CPU/GPU.
- **[NeMo AutoModel — dLLM Fine-Tuning](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dllm-fine-tuning.html)** — NVIDIA recipes for SFT/LoRA of LLaDA/LLaDA2, Nemotron-Labs-Diffusion, DiffusionGemma, DFlash and AR-to-diffusion conversions.
- **[LMMs-Engine](https://github.com/EvolvingLMMs-Lab/lmms-engine)** — Unified multimodal training engine with a `dllm_trainer` for masked diffusion LMs (Qwen3-based) using FSDP2/DeepSpeed and Muon.
- **[nanoDiff](https://github.com/BY571/nanoDiff)** — Minimal nanoGPT-style PyTorch masked diffusion LM following the LLaDA recipe, with pretraining, SFT and a 50M–350M scaling family.
- **[Awesome-Diffusion-LLM (awesome-list)](https://github.com/yczhou001/Awesome-Diffusion-LLM)** — Paper list, tutorial and nano code notebooks (continuous and masked diffusion) for diffusion LLMs.
- **[DLLM-Survey (awesome-list)](https://github.com/LiQiiiii/DLLM-Survey)** — Companion paper collection for the TMLR'26 survey "Discrete Diffusion in Large Language and Multimodal Models", including a toolkits section.

### Datasets & benchmarks

- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — standard pretraining / perplexity corpus.
- [LM1B / One Billion Word](https://huggingface.co/datasets/billion-word-benchmark/lm1b) — sentence-level NLL benchmark.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) ([source](https://github.com/openai/grade-school-math)) — grade-school math reasoning.
- [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) — Python coding benchmark.
- **[ParallelBench](https://github.com/furiosa-ai/ParallelBench)** — ICLR'26 benchmark exposing quality loss under parallel decoding in dLLMs, with a public [leaderboard](https://parallelbench.github.io/leaderboard/).
- **[dLLM Leaderboard (AUP)](https://huggingface.co/spaces/d3LLM/dLLM_Leaderboard)** — HF Space ranking diffusion LLMs on five tasks by Accuracy Under Parallelism (AUP), a hardware-independent speed–accuracy metric.

### Blogs & talks

- [Diffusion language models — Sander Dieleman](https://sander.ai/2023/01/09/diffusion-language.html) — best conceptual intro.
- [Language Modeling by Estimating Ratios — Aaron Lou (SEDD)](https://aaronlou.com/blog/2024/discrete-diffusion/)
- [Score-based generative modeling — Yang Song](https://yang-song.net/blog/2021/score/)
- [Dream 7B](https://hkunlp.github.io/blog/2025/dream/) · [Dream-Coder](https://hkunlp.github.io/blog/2025/dream-coder/) · [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/) — HKU NLP.
- [Introducing Mercury — Inception Labs](https://www.inceptionlabs.ai/blog/introducing-mercury)
- [Gemini Diffusion — Google](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/)
- **[How to Build a Diffusion Language Model — Kuleshov Group](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/)** — 2026 tutorial (from ICLR/MLSS 2026 lectures) covering masked/block diffusion, remasking, distillation, guidance and RL post-training.
- **[Discrete Diffusion Reading Group](https://d-llms.io/)** — Weekly reading group (Sahoo, Deschenaux, Yang) on discrete/language diffusion with recorded sessions on YouTube.
- **[Day-0 Support for LLaDA 2.0 in SGLang — LMSYS](https://lmsys.org/blog/2025-12-19-diffusion-llm)** — Dec 2025 post on adding dLLM serving to SGLang by reusing chunked prefill, with LLaDA2.0 benchmarks.
- **[DiffusionGemma in vLLM — vLLM blog](https://vllm.ai/blog/2026-06-10-diffusion-gemma)** — Jun 2026 post on the first dLLM natively supported in vLLM via Model Runner v2's ModelState and the speculative-decoding path.
- **[DiffusionGemma: 4x faster text generation — Google](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)** — Google's Jun 2026 launch post for the experimental open-weight Gemma 4-based text diffusion model.
- **[DiffusionGemma: How to Run Locally — Unsloth](https://unsloth.ai/docs/models/diffusiongemma)** — Practical guide to running quantized DiffusionGemma locally via Unsloth Studio or llama.cpp's `llama-diffusion-cli`.
- **[Introducing Mercury 2 — Inception](https://www.inceptionlabs.ai/blog/introducing-mercury-2)** — Launch of Mercury 2, billed as the first reasoning dLLM (~1,000 tok/s on Blackwell).
- **[Introducing Mercury 3 — Inception](https://www.inceptionlabs.ai/blog/introducing-mercury-3)** — Oct 2026 launch of Mercury 3 with 1M-token context, >1,000 tok/s, and availability via API, OpenRouter, Baseten and Vercel.
- **[AUP: when Accuracy Meets Parallelism — Hao AI Lab](https://haoailab.com/blogs/text-diffusion/)** — Dec 2025 post arguing open dLLMs trade accuracy for parallelism and introducing the AUP metric and d3LLM.
- **[Seed Diffusion Preview — ByteDance Seed](https://seed.bytedance.com/en/blog/seed-research-seed-diffusion-preview-released-a-diffusion-language-model-delivering-breakthrough-2-146-tokens-s-inference-speed)** — Jul 2025 announcement of Seed Diffusion Preview, a code dLLM reaching 2,146 tok/s on H20 GPUs.

### Demos

- [Mercury playground](https://chat.inceptionlabs.ai) · [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [LLaDA Space](https://huggingface.co/spaces/multimodalart/LLaDA) · [LLaDA demo page](https://ml-gsai.github.io/LLaDA-demo/) · [MMaDA Space](https://huggingface.co/spaces/Gen-Verse/MMaDA)
- **[Seed Diffusion Preview demo](https://studio.seed.ai/exp/seed_diffusion/)** — Official ByteDance Seed web demo for trying Seed Diffusion Preview.
- **[Inception Platform API docs](https://docs.inceptionlabs.ai/)** — OpenAI-compatible API docs for Mercury 3, Mercury 2.5 and Mercury Edit 2 diffusion models.
- **[Dream-Coder Space](https://huggingface.co/spaces/ZiruiWu/Dream-Coder-Instruct-7B)** — Hugging Face ZeroGPU demo of Dream-Coder-v0-Instruct-7B.
- **[DreamOn Space](https://huggingface.co/spaces/ZiruiWu/DreamOn-v0-7B)** — Hugging Face ZeroGPU demo of DreamOn-v0-7B variable-length code infilling.

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
9. **2026 H2: open frontier dLLMs and editing.** Google's open-weight **DiffusionGemma** (served by vLLM, SGLang and llama.cpp), NVIDIA's tri-mode **Nemotron-Labs-Diffusion**, **iLLaDA**, and **LLaDA 2.1/2.2** (token editing → insert/delete edits for agents); Mercury 3 reaches 1M context. The Oct 2026 refresh adds 268 papers (§19), concentrated in RL post-training, KV-cache/parallel-decoding systems, AR→diffusion conversion, and multimodal/VLA dLLMs.

---

## Contributing

PRs welcome! Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** — real links only, verify arXiv IDs, one factual line per entry, no duplicates.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](LICENSE)

To the extent possible under law, contributors have waived all copyright and related or neighboring rights to this work ([CC0 1.0](LICENSE)).
