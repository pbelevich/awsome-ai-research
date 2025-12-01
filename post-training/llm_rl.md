# LLM Reinforcement Learning

## 1. For LLM work, “Reinforcement Learning” is mostly:

### RLHF / RLAIF-style pipelines

1. SFT on instruction data.
2. Reward model trained on preference data.
3. RL step (PPO, GRPO, etc.) where the actor is optimized against the reward model, often with a reference model for KL control.
    1. https://docs.nvidia.com/nemo-framework/user-guide/24.07/modelalignment/index.html

### Preference-based variants (no explicit RL):
* DPO, IPO, RPO, etc., which are implemented in TRL and NeMo-Aligner.
    * https://aclanthology.org/2025.emnlp-demos.48/
### Algorithms
* PPO / GRPO / REINFORCE++ in TRL, Verl, OpenRLHF, NeMo RL.
* DPO / (R)PO etc. in TRL & NeMo RL / NeMo-Aligner.
* https://aclanthology.org/2025.emnlp-demos.48/

## 2. Quick snapshot of each framework

### 2.1 TRL (Hugging Face) https://huggingface.co/docs/trl/en/index

* Full-stack post-training library for LLMs: SFT, GRPO, DPO, reward modeling, PPO-style RLHF.
* Integrates tightly with Transformers and Accelerate, so you get DDP, FSDP, DeepSpeed, etc. out of the box.
    * https://github.com/huggingface/trl
* Works with vLLM integration paths as well.
    * https://docs.vllm.ai/en/latest/training/trl/
* No native Ray integration, but nothing stops you from:
    * Running TRL jobs inside Ray Tasks/Actors, or
    * Just using Slurm directly for multi-node SFT/RLHF.

FP8 / long-context angle:
Via Accelerate, you can use mixed_precision="fp8" with either TransformerEngine or MS-AMP backends, which TRL can inherit. https://huggingface.co/docs/accelerate/v0.34.1/en/usage_guides/low_precision_training
Long context is just whatever your base model / backend (e.g. Llama-3-8B-128k) supports.

### 2.2 Verl https://github.com/volcengine/verl

* Verl (“Volcano Engine RL for LLMs”): flexible, production-grade RLHF / RL framework for LLMs.
* Designed specifically for LLM RLHF pipelines (actor, critic, reward, reference) with scalable scheduling (HybridFlow).
* Supports PPO / GRPO / DAPO and integrates with vLLM, SGLang, HF models, and distributed backends like FSDP / Megatron.
* Ray-native and has official KubeRay example for RLHF on GSM8K with Qwen2.5-0.5B-Instruct.
    * https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html

FP8 / long-context angle:
Because Verl sits on top of PyTorch, FSDP, Megatron, etc., you can layer TransformerEngine FP8 and Megatron long-context configs underneath (similar to NeMo RL; see below).
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html

### 2.3 OpenRLHF https://github.com/OpenRLHF/OpenRLHF

* OpenRLHF: Ray + vLLM + DeepSpeed + HF Transformers RLHF framework.
* Built to scale 70B+ full RLHF by decoupling actor / critic / reward / reference with Ray-based schedulers and vLLM inference. https://arxiv.org/abs/2405.11143
* Supports PPO, GRPO, REINFORCE++, DPO, rejection sampling, async agentic RL.
    * https://github.com/OpenRLHF/OpenRLHF

FP8 / long-context angle:

* Roadmap + issues show FP8 support for large models (e.g. DeepSeek FP8) using SGLang / quantization.
    * https://github.com/OpenRLHF/OpenRLHF/issues/568
* Long context comes via vLLM / underlying model (Qwen, Llama-3, etc.), which OpenRLHF already supports.
    * https://blog.vllm.ai/2025/04/23/openrlhf-vllm.html

OpenRLHF is your most Ray-heavy playground.

### 2.4 NeMo RL / NeMo-Aligner

* NeMo RL: NVIDIA’s RL post-training library (GRPO, PPO, DPO, reward modeling, on-policy distillation, etc.). Uses Megatron-Core / DTensor for high-performance training and Ray for orchestration.
    * https://docs.nvidia.com/nemo/rl/latest/index.html
* Comes with integrated generation backends: vLLM and Megatron-native inference.
    * https://github.com/NVIDIA-NeMo/RL
* NeMo docs explicitly show how to run NeMo RL with Ray on Slurm or Kubernetes.
    * https://docs.nvidia.com/nemo/rl/latest/cluster.html

Long-context & FP8:

* NeMo provides long-context training recipes (Llama, Mixtral, Nemotron) and blog posts on scaling to millions of tokens.
    * https://docs.nvidia.com/nemo-framework/user-guide/latest/longcontext/index.html
* NeMo RL has dedicated FP8 quantization support, using TransformerEngine FP8 lin layers and DeepSeek-style scaling.
    * https://docs.nvidia.com/nemo/rl/latest/fp8.html
* Megatron-Core flags include --fp8-hybrid for RL training.
    * https://pypi.org/project/megatron-core/0.16.0rc0.dev128634/

This is the most natural place to combine RL + long context + FP8 in a “clean” supported way.

## 3. Orchestration patterns: Ray, Kubernetes, Slurm

### Ray on Slurm

* Ray docs show templates and symmetric run scripts to start a Ray cluster inside a Slurm job and then run Python entrypoints on the head node - https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html

### Ray on Kubernetes (KubeRay)

* KubeRay operator manages RayCluster / RayJob / RayService CRDs on K8s.
    * https://docs.ray.io/en/latest/cluster/kubernetes/index.html
* Verl and OpenRLHF both have KubeRay–oriented docs / examples.
    * https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html

### Pure Slurm

* For TRL (and potentially NeMo RL via NeMo-Run), you can stay in the “classic” Slurm world and optionally let NeMo RL internally spin up Ray (their cluster docs show exactly that).
    * https://docs.nvidia.com/nemo/rl/latest/cluster.html

