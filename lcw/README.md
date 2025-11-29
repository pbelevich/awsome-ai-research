# Long context training

## 1. What “long-context training” is really about

### 1.1. Goals

When people say long-context training they’re usually after one or more of:

* Extend context of an existing model (e.g. Llama-2 4k -> 16k/32k; Llama-3.1 128k -> 1M+).
* Train a model from scratch with long context (e.g. 16k/32k) – less common because it’s expensive.
* Enable effective use of long context, not just “accept more tokens” (i.e. actually retrieve and reason over information far away in the sequence).

Typical target windows now:

* “Medium-long”: 16k–32k
* “Long”: 64k–128k
* “Ultra-long”: 128k–4M (e.g. Nemotron-UltraLong 1M/2M/4M) - https://research.nvidia.com/labs/adlr/ultralong/

### 1.2. The core difficulty

Compared to 2k–4k context, long-context training adds:

* Quadratic attention cost O(L^2 * d) – both FLOPs and memory.
* Huge KV + activations footprint – even with FlashAttention; context parallelism or sequence parallelism becomes almost mandatory beyond ~16–32k. 
* Optimization quirks: RoPE / PE has to be modified; naïve BF16 + RoPE can break at long lengths.
    * https://openreview.net/pdf/ae0344247c9b0d65f31772d9aaa7145da953f118.pdf
* Data + curriculum: You need long documents and a careful length distribution; long sequences are rare and noisy on the raw web. 

So long-context training is really: A combination of continued pre-training, positional encoding tricks, data curation, and system tricks (CP/SP + checkpointing + quantization).

## 2. Key algorithmic recipes

### 2.1. Continued pretraining with longer sequences

This is the dominant pattern.

Meta’s “Effective Long-Context Scaling” (Llama-2-Long)

* Start from Llama-2 checkpoints.
* Change only RoPE base (smaller rotation base -> slower phase growth -> less decay with distance).
* Continue pretraining for ~400B tokens at seq length 16k–32k, keeping tokens per batch constant (fewer sequences per batch as seq length grows).
* Use FlashAttention so memory overhead is small; 4k → 16k only ~17% slowdown for 70B. 

NVIDIA Nemotron-UltraLong (Llama-3.1-8B -> 1M/2M/4M)

* Continue pretraining an 8B instruct model for only ≈1B tokens at ultra-long contexts (1M, 2M, 4M).
* Upsample long documents; concatenate docs with special separators and disable cross-doc masks so the model can attend across the full mega-sequence.
* Use YaRN-style RoPE scaling to safely stretch to millions of tokens.
    * https://research.nvidia.com/labs/adlr/ultralong/

General pattern (and what your examples should show):

1. Base model pretraining at 2k–4k.
2. Continued pretraining at 16k.
3. Possibly another stage at 32k/64k or 128k+.
    1. https://docs.nvidia.com/nemo-framework/user-guide/latest/longcontext/index.html

### 2.2. “Train-short, test-long” / efficient recipes

Instead of actually training at the full target context:

* GeNE (Generalized Extrapolation for RoPE):
    Randomly vary the RoPE interpolation/scale factor batch-by-batch, so the model learns to generalize to much longer positions than seen in training.
    * https://aclanthology.org/2024.findings-acl.249.pdf
* LongRecipe:
    Shows you can get to 128k effective context while only ever training with sequences up to ~30% of that length (e.g. ~38k tokens), saving ~85% of the compute. They do:
    * “Impactful token analysis” to pick where long-range information matters,
    * Position index transformations (simulating long contexts),
    * A tuned training curriculum
        * https://aclanthology.org/2025.acl-long.581.pdf
* LongRoPE2 / HARPE and similar works:
    More sophisticated RoPE modifications or head-wise scaling; typically single-stage continued pretraining with a clever PE so you don’t need multi-stage context schedules.
    * https://arxiv.org/pdf/2502.20082

### 2.3. RoPE & PE tricks

Most long-context models today still use full attention with RoPE:

* Simple RoPE base change (Llama-2-Long). 
* Positional interpolation / NTK-aware interpolation – scale position indices non-linearly to compress long contexts. 
    * https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html
* YaRN: mixtures of multiple interpolation regimes + modest continued pretraining to reach 32k+.
    * https://arxiv.org/pdf/2309.00071
* LongRoPE2: near-lossless context scaling using learned or searched scaling schedules and “needle-driven” evaluation for calibration.
    * https://arxiv.org/pdf/2502.20082

Your examples can show:

* Pure RoPE base change,
* RoPE scaling configs (rope_scaling={"type": "yarn", ...} in HF),
* Possibly a LongRecipe/GeNE-style “dynamic scale”.

### 2.4. Data + curriculum

Across Meta Llama-2-Long, UltraLong, ProLong, etc., the main ideas repeat: 

* Use high-quality, long documents (legal, scientific, books, code, etc.).
* Control length distribution: include a mix of short, medium, and long docs; don’t train only on extremely long sequences.
* Pack documents with <eod> tokens and either:
    * keep cross-doc mask (classical LM), or
    * deliberately disable it when trying to teach cross-doc retrieval (UltraLong).
* Curriculum: gradually increase max sequence length; some recipes also increase the fraction of long sequences over time.

Practical open datasets you can base examples on:

* SlimPajama / SlimPajama-DC – deduplicated multi-source corpus (627B tokens) with tooling.
    * https://huggingface.co/datasets/cerebras/SlimPajama-627B
* FineWeb / FineWeb-Edu – high-quality cleaned web corpus, plus an “educational” filtered subset. Great source of long, coherent docs.
    * https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

You don’t need to implement a new dataset; just show how to pack & sample it for long context.

### 2.5. Evaluation & what “good” means

Better recipes evaluate on real tasks, not just perplexity or one Needle test: 

* Benchmarks: LongBench, ZeroScrolls, RULER, multi-doc QA, summarization, retrieval tasks.
* Synthetic: Needle-in-a-Haystack, long copy/recall tasks – still useful sanity checks.

You probably want at least one SLURM eval script per example, even if your immediate ask is “training”.

## 3. System + parallelism patterns you should assume

### 3.1. Baseline: DDP/FSDP + FlashAttention

For 16k–32k on 7–8B models you can often get away with:

* FSDP / ZeRO-3 for parameters + optimizer states.
* FlashAttention-2 / SDPA for attention.
* Activation checkpointing and maybe CPU/NVMe offload. 

### 3.2. Sequence parallelism (DeepSpeed-Ulysses)

DeepSpeed-Ulysses:

* Shards the sequence dimension across GPUs with an all-to-all to compute attention.
* Supports up to 1M tokens on 64 A100s in the original tutorial.
    * https://www.deepspeed.ai/tutorials/ds-sequence/
* New HF integration + ALST (Arctic Long Sequence Training) adds:
    * Ulysses SP for HF Transformers,
    * Tiled loss, tiled MLP,
    * Activation checkpoint offload to CPU,
    * Recommended PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
        * https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/

This is perfect for Slurm examples that use deepspeed or torchrun on a small cluster.

### 3.3. Context parallelism (Megatron / NeMo / NeuronX)

Context Parallelism (CP) (Megatron-LM, NeMo, AWS SMP, NeuronX-Distributed):

* Also splits activations along the sequence dimension, but tightly integrated into the model’s parallel groups.
    * https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html
* Used in NeMo long-context recipes for Llama-3 and Nemotron, e.g. a staged pipeline:
    * Base pretrain at 2k,
    * 16k stage with CP,
    * 64k stage with higher CP degree.
        * https://docs.nvidia.com/nemo-framework/user-guide/latest/longcontext/index.html

NeMo has built-in Slurm support via --slurm and recipe configs, which is ideal for your “canonical” SLURM examples.
https://docs.nvidia.com/nemo-framework/user-guide/latest/longcontext/index.html

### 3.4. Extra tricks: FP8, tiled MLP, offload

* AWS SMP + FP8 examples show: 16k context with context-parallel + FP8 is much more memory-efficient than BF16 without CP.
    * https://aws.amazon.com/blogs/machine-learning/efficiently-train-models-with-large-sequence-lengths-using-amazon-sagemaker-model-parallel/
* DeepSpeed ALST shows the “bag of tricks” to push Llama-8B to multi-million tokens using Ulysses SP + tiled loss/MLP + CPU offloaded checkpoints.
    * https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/

## Links
* DeepSpeed-Ulysses: https://www.deepspeed.ai/tutorials/ds-sequence/ + Paper: https://arxiv.org/pdf/2309.14509
* Qwen2.5-1M: https://qwenlm.github.io/blog/qwen2.5-1m/
* [Princeton long-context language models](https://github.com/princeton-nlp/ProLong) + [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/pdf/2410.02660)
* [Arctic Long Sequence Training (ALST): Scalable And Efficient Training For Multi-Million Token Sequences](https://www.snowflake.com/en/engineering-blog/arctic-long-sequence-training-multi-million-token-ai/) + [Paper](https://arxiv.org/pdf/2506.13996)
* [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](https://arxiv.org/pdf/2410.02694)
