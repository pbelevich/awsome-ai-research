# Llama 4 Scout on vLLM

https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama4-Scout.html

```bash
mkdir -p docker
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/docker/Dockerfile -O docker/Dockerfile
```

```bash
./build_image.sh
```

```bash
./run_container.sh
```

## FP8

Download the model:
```bash
hf download nvidia/Llama-4-Scout-17B-16E-Instruct-FP8
```

Launch on B200:
```bash
vllm serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8080 \
  --tokenizer nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}' \
  --enable-chunked-prefill \
  --async-scheduling \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 1 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --max-model-len 9216
```

Test the model:
```bash
curl http://${HOST}:${PORT}/v1/completions -H "Content-Type: application/json" -d '{ "model": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8", "prompt": "San Francisco is a", "max_tokens": 20, "temperature": 0 }' | jq
```

```json
{
  "id": "cmpl-c03b898b5d8a485fadebc1b13554a8c4",
  "object": "text_completion",
  "created": 1754949942,
  "model": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
  "choices": [
    {
      "index": 0,
      "text": " city known for its vibrant culture, stunning architecture, and iconic landmarks. One of the most recognizable symbols",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 25,
    "completion_tokens": 20,
    "prompt_tokens_details": null
  },
  "kv_transfer_params": null
}
```

Evaluation:
```bash
lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args \
base_url=http://${HOST}:${PORT}/v1/completions,\
model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8,\
tokenized_requests=False,tokenizer_backend=None,\
num_concurrent=128,timeout=120,max_retries=5
```

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9219|±  |0.0074|
|     |       |strict-match    |     5|exact_match|↑  |0.9075|±  |0.0080|

Benchmarking Performance

```bash
python3 /vllm-workspace/benchmarks/benchmark_serving.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --ignore-eos \
  --max-concurrency 512 \
  --num-prompts 2560 \
  --save-result --result-filename vllm_benchmark_serving_results.json
```

```
============ Serving Benchmark Result ============
Successful requests:                     2560
Maximum request concurrency:             512
Benchmark duration (s):                  654.98
Total input tokens:                      2616855
Total generated tokens:                  2621440
Request throughput (req/s):              3.91
Output token throughput (tok/s):         4002.32
Total Token throughput (tok/s):          7997.65
---------------Time to First Token----------------
Mean TTFT (ms):                          21437.04
Median TTFT (ms):                        13619.42
P99 TTFT (ms):                           88099.07
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          102.25
Median TPOT (ms):                        90.17
P99 TPOT (ms):                           163.85
---------------Inter-token Latency----------------
Mean ITL (ms):                           102.25
Median ITL (ms):                         66.39
P99 ITL (ms):                            375.98
==================================================
```

## FP4

Download the model:
```bash
hf download nvidia/Llama-4-Scout-17B-16E-Instruct-FP4
```

Launch on B200:
```bash
vllm serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP4 \
  --host 0.0.0.0 \
  --port 8080 \
  --tokenizer nvidia/Llama-4-Scout-17B-16E-Instruct-FP4 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}' \
  --enable-chunked-prefill \
  --async-scheduling \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 1 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --max-model-len 9216
```

Test the model:
```bash
curl http://${HOST}:${PORT}/v1/completions -H "Content-Type: application/json" -d '{ "model": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4", "prompt": "San Francisco is a", "max_tokens": 20, "temperature": 0 }' | jq
```

```json
{
  "id": "cmpl-802331736a0b4360bc581db64f513ed1",
  "object": "text_completion",
  "created": 1754949951,
  "model": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
  "choices": [
    {
      "index": 0,
      "text": " city known for its vibrant culture, stunning architecture, and breathtaking views. One of the best ways to",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 25,
    "completion_tokens": 20,
    "prompt_tokens_details": null
  },
  "kv_transfer_params": null
}
```

Evaluation:
```bash
lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args \
base_url=http://${HOST}:${PORT}/v1/completions,\
model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP4,\
tokenized_requests=False,tokenizer_backend=None,\
num_concurrent=128,timeout=120,max_retries=5
```

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8969|±  |0.0084|
|     |       |strict-match    |     5|exact_match|↑  |0.8779|±  |0.0090|

Benchmarking Performance

```bash
python3 /vllm-workspace/benchmarks/benchmark_serving.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP4 \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --ignore-eos \
  --max-concurrency 512 \
  --num-prompts 2560 \
  --save-result --result-filename vllm_benchmark_serving_results.json
```

```
============ Serving Benchmark Result ============
Successful requests:                     2560
Maximum request concurrency:             512
Benchmark duration (s):                  385.57
Total input tokens:                      2616855
Total generated tokens:                  2621440
Request throughput (req/s):              6.64
Output token throughput (tok/s):         6798.82
Total Token throughput (tok/s):          13585.75
---------------Time to First Token----------------
Mean TTFT (ms):                          2847.42
Median TTFT (ms):                        1923.98
P99 TTFT (ms):                           12680.62
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          70.11
Median TPOT (ms):                        70.82
P99 TPOT (ms):                           77.64
---------------Inter-token Latency----------------
Mean ITL (ms):                           70.11
Median ITL (ms):                         62.31
P99 ITL (ms):                            215.29
==================================================
```
