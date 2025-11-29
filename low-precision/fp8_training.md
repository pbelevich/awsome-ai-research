# FP8 training

## 1. What “FP8 training” really means

In practice not everything runs in FP8:

* FP8 formats
    * E4M3: more mantissa, less range – usually for activations & weights.
    * E5M2: more range, less mantissa – usually for gradients or backward.
        Most stacks use “HYBRID”: E4M3 in fwd, E5M2 in bwd.
    * https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
* Recipes (how scaling is done) — examples from TransformerEngine & Megatron:
    * tensorwise / DelayedScaling: per-tensor scales with amax-history and delayed updates.
    * MXFP8 / blockwise: block-wise scaling for better robustness at some extra overhead.
* Where FP8 is used
    * GEMMs in MLP & attention -> FP8 GEMM + FP16/BF16 accumulation.
    * LayerNorm, softmax, reductions, optimizer updates -> still FP16/BF16/FP32.
    * https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-0.8.0/user-guide/
* Common training patterns
    * Master weights in FP16/BF16 or FP32; FP8 views for compute.
    * Loss scaling or delayed scaling recipes to avoid underflow/overflow.
    * Sometimes gradients, optimizer states, and even all-reduce comms go FP8 (MS-AMP / FP8-LM style).
    * MS-AMP paper https://arxiv.org/pdf/2310.18313
* Performance expectations
    * Vendor numbers: up to ~2x speedup vs BF16 on GEMM-heavy workloads.
        * https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/
    * Real Megatron reports often ~30–50% end-to-end speedup depending on model, parallelism and input pipeline.
        * https://github.com/NVIDIA/Megatron-LM/issues/396
* Stability
    * TE/Accelerate/Megatron default recipes are designed to be accuracy-parity or close vs BF16, with minimal or no hyper-param tuning.

## 2. Main FP8 software stacks

### 2.1 NVIDIA TransformerEngine (TE) - cleanest way to build custom PyTorch FP8 examples

Core library for Hopper/Blackwell FP8:

* Drop-in FP8 versions of Linear, LayerNorm, transformer blocks, etc.
* fp8_autocast context + “recipes” (DelayedScaling, MXFP8BlockScaling, etc.) to control scaling & formats.
* Native support for DP, TP, sequence parallelism and optimizations like FP8 weight caching for GA steps.
* https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html

### 2.2 Megatron-LM / NeMo Megatron-Bridge - serious, production-style FP8 LLM training

Megatron/Core + NeMo Megatron-Bridge wire TE into full LLM training:

* Flags like --fp8-format e4m3|hybrid and --fp8-recipe tensorwise|delayed|mxfp8|blockwise. 
    * https://swift.readthedocs.io/en/v3.7/Instruction/Megatron-SWIFT-Training.html
* TE-based attention & MLP; can use cuDNN FlashAttention with FP8 kernels.
    * https://github.com/NVIDIA/Megatron-LM
* Integrates with DP/TP/PP/FSDP; Megatron-Bridge also supports FP8 parameter AllGather to reduce comm cost.
    * https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html

### 2.3 HF Accelerate + FP8 (TE / MS-AMP / TorchAO) - “HF-native” FP8 Slurm examples

Hugging Face Accelerate exposes FP8 in a framework-friendly way:

* accelerate launch --mixed_precision fp8 ... to enable FP8 end-to-end (with Trainer too)
    * https://discuss.huggingface.co/t/trainer-with-fp8-what-to-use-in-accel-cli-vs-trainingarguments/79359
* Under the hood it can use:
    * TransformersEngine (TE) to auto-replace nn.Linear/nn.LayerNorm with FP8 versions.
        * https://huggingface.co/docs/accelerate/en/concept_guides/low_precision_training
    * MS-AMP (Microsoft) for deeper FP8 usage: grads, optimizer states, comms, with “O1/O2” optimization levels.
        * https://azure.github.io/MS-AMP/docs/user-tutorial/usage/
* Accelerate ships FP8 examples for single-GPU, DDP, FSDP, and DeepSpeed ZeRO1–3.
    * https://huggingface.co/docs/accelerate/en/usage_guides/low_precision_training

### 2.4 FP8-LM style “full-stack FP8”

The FP8-LM paper pushes FP8 deeper:

* Uses FP8 for weights, activations, gradients, optimizer states and distributed training in an incremental way.
* Reports ~39% memory reduction and ~75% training speedup vs BF16 on large LLMs.
* Built atop Megatron-LM; conceptually similar to MS-AMP O1/O2.

### 2.5 Caveats: DeepSpeed + FP8

* FP8 can be used with DeepSpeed via Accelerate / MS-AMP, but HF issues report sharp edges:
    * e.g., enabling FP8 with some configs silently disables ZeRO stages or behaves unexpectedly.
        * https://github.com/huggingface/accelerate/issues/3360
* Good target for an “advanced / experimental” SLURM example where you explicitly warn about limitations.

