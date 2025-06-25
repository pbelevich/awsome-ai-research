# Estimations

## LLM Training Estimations

FLOPs_per_token = 8 * model_size (with activation checkpointing) or 6 * model_size (without activation checkpointing)

Compute_budget = FLOPs_per_token * dataset_size

Effective_GPU_dtype_flop_per_second = HFU * GPU_dtype_flop_per_second

GPU*hours = Compute_budget / Effective_GPU_dtype_flop_per_second = (FLOPs_per_token * dataset_size) / (HFU * GPU_dtype_flop_per_second) / 3600 = (8 * model_size * dataset_size) / (HFU * GPU_dtype_flop_per_second * 3600)

### Example Meta LLama 3 405B pretraining on 15.6T tokens dataset with bf16 mixed precision with activation checkpointing on NVIDIA H100 GPUs:

FLOPs_per_token = 8 * 405e9 = 3.24e12 = 3.24 TFLOPs/token (forward: 810 GFLOPs/token and backward: 1620 GFLOPs/token)

Compute_budget = FLOPs_per_token * dataset_size = 8 * model_size * dataset_size = 8 * 405B * 15.6T = 5e25 FLOPs

Assuming that HFU is 45% and [H100 bf16 has 1PFLOP/s](https://www.nvidia.com/en-us/data-center/h100/)

GPU*hours = (8 * 405e9 * 15.6e12) / (0.45 * 1e15) / 3600 = [31M GPU * hours](https://huggingface.co/meta-llama/Llama-3.1-405B)

Taking into account that it was trained on 16K H100 it translates to 31M / (16384 H100) / 24h = 80 days
