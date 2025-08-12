# gpt-oss on vLLM

* https://blog.vllm.ai/2025/08/05/gpt-oss.html
* https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#b200
* https://cookbook.openai.com/articles/gpt-oss/run-vllm

```bash
# All 3 of these are required
export VLLM_USE_TRTLLM_ATTENTION=1
export VLLM_USE_TRTLLM_DECODE_ATTENTION=1
export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1

# Pick only one out of the two.
# mxfp8 activation for MoE. faster, but higher risk for accuracy.
export VLLM_USE_FLASHINFER_MXFP4_MOE=1 
# bf16 activation for MoE. matching reference precision.
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1 
```

## `gpt-oss-20b`

Download the model:
```bash
hf download openai/gpt-oss-20b
```

Launch on B200:
```bash
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    -v $HF_HOME:/root/.cache/huggingface \
    -e VLLM_USE_TRTLLM_ATTENTION=1 \
    -e VLLM_USE_TRTLLM_DECODE_ATTENTION=1 \
    -e VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1 \
    -e VLLM_USE_FLASHINFER_MXFP4_MOE=1 \
    vllm/vllm-openai:gptoss \
    --host 0.0.0.0 --port 8000 \
    --model openai/gpt-oss-20b --async-scheduling
```

List available models:
```bash
curl http://${HOST}:${PORT}/v1/models | jq
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/gpt-oss-20b",
      "object": "model",
      "created": 1754872339,
      "owned_by": "vllm",
      "root": "openai/gpt-oss-20b",
      "parent": null,
      "max_model_len": 131072,
      "permission": [
        {
          "id": "modelperm-2aa14dd031a64fff879aba68002abde2",
          "object": "model_permission",
          "created": 1754872339,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

Test the model:
```bash
curl -s http://${HOST}:${PORT}/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }' | jq
```

```json
{
  "id": "chatcmpl-7d3fbdf9ed0c4c019cb947a88f0cd62a",
  "object": "chat.completion",
  "created": 1754872507,
  "model": "openai/gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning_content": "User asks a simple question: \"What is the capital of France?\" We respond with the answer: Paris."
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 78,
    "total_tokens": 117,
    "completion_tokens": 39,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```

Evaluation:

```bash
conda create -yn gpt-oss python=3.12
conda activate gpt-oss
pip install "gpt-oss @ git+https://github.com/openai/gpt-oss.git"
```

GPQA:

```bash
OPENAI_API_KEY= python -m gpt_oss.evals --base-url http://${HOST}:${PORT}/v1 --model openai/gpt-oss-20b --eval gpqa --n-threads 128
```
```
[{'eval_name': 'gpqa', 'model_name': 'openai/gpt-oss-20b-low_temp1.0_20250811_141332', 'metric': 0.5612373737373737}, {'eval_name': 'gpqa', 'model_name': 'openai/gpt-oss-20b-medium_temp1.0_20250811_141332', 'metric': 0.63510101010101}, {'eval_name': 'gpqa', 'model_name': 'openai/gpt-oss-20b-high_temp1.0_20250811_141332', 'metric': 0.7209595959595959}]
```

AIME25:

```bash
OPENAI_API_KEY= python -m gpt_oss.evals --base-url http://${HOST}:${PORT}/v1 --model openai/gpt-oss-20b --eval aime25 --n-threads 128
```
```
[{'eval_name': 'aime25', 'model_name': 'openai/gpt-oss-20b-low_temp1.0_20250811_191800', 'metric': 0.35}, {'eval_name': 'aime25', 'model_name': 'openai/gpt-oss-20b-medium_temp1.0_20250811_191800', 'metric': 0.7458333333333333}, {'eval_name': 'aime25', 'model_name': 'openai/gpt-oss-20b-high_temp1.0_20250811_191800', 'metric': 0.8625}]
```

Model: 20B
| Reasoning Effort | GPQA | AIME25 |
|------------------|------|--------|
| Low              | 56.1 |   35.0 |
| Medium           | 63.5 |   74.6 |
| High             | 72.1 |   86.3 |

## `gpt-oss-120b`

Download the model:
```bash
hf download openai/gpt-oss-120b
```

Launch on B200:
```bash
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    -v $HF_HOME:/root/.cache/huggingface \
    -e VLLM_USE_TRTLLM_ATTENTION=1 \
    -e VLLM_USE_TRTLLM_DECODE_ATTENTION=1 \
    -e VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1 \
    -e VLLM_USE_FLASHINFER_MXFP4_MOE=1 \
    vllm/vllm-openai:gptoss \
    --host 0.0.0.0 --port 8000 \
    --model openai/gpt-oss-120b --async-scheduling --tensor-parallel-size 8
```

List available models:
```bash
curl http://${HOST}:${PORT}/v1/models | jq
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/gpt-oss-120b",
      "object": "model",
      "created": 1754877689,
      "owned_by": "vllm",
      "root": "openai/gpt-oss-120b",
      "parent": null,
      "max_model_len": 131072,
      "permission": [
        {
          "id": "modelperm-deb90cdda75642ff92606784e1368350",
          "object": "model_permission",
          "created": 1754877689,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

Test the model:
```bash
curl -s http://${HOST}:${PORT}/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }' | jq
```

```json
{
  "id": "chatcmpl-fa9600be8833416a818ea73fb29c52ce",
  "object": "chat.completion",
  "created": 1754879240,
  "model": "openai/gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is **Paris**.",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning_content": "The user asks a simple question: What is the capital of France? Answer: Paris.\n\nWe just respond."
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 78,
    "total_tokens": 119,
    "completion_tokens": 41,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```

Evaluation:

```bash
conda create -yn gpt-oss python=3.12
conda activate gpt-oss
pip install "gpt-oss @ git+https://github.com/openai/gpt-oss.git"
```

GPQA:

```bash
OPENAI_API_KEY= python -m gpt_oss.evals --base-url http://${HOST}:${PORT}/v1 --model openai/gpt-oss-120b --eval gpqa --n-threads 128
```
```
[{'eval_name': 'gpqa', 'model_name': 'openai/gpt-oss-120b-low_temp1.0_20250811_142915', 'metric': 0.6496212121212122}, {'eval_name': 'gpqa', 'model_name': 'openai/gpt-oss-120b-medium_temp1.0_20250811_142915', 'metric': 0.7032828282828283}, {'eval_name': 'gpqa', 'model_name': 'openai/gpt-oss-120b-high_temp1.0_20250811_142915', 'metric': 0.7929292929292929}]
```

AIME25:

```bash
OPENAI_API_KEY= python -m gpt_oss.evals --base-url http://${HOST}:${PORT}/v1 --model openai/gpt-oss-120b --eval aime25 --n-threads 128
```
```
[{'eval_name': 'aime25', 'model_name': 'openai/gpt-oss-120b-low_temp1.0_20250811_194713', 'metric': 0.525}, {'eval_name': 'aime25', 'model_name': 'openai/gpt-oss-120b-medium_temp1.0_20250811_194713', 'metric': 0.7416666666666667}, {'eval_name': 'aime25', 'model_name': 'openai/gpt-oss-120b-high_temp1.0_20250811_194713', 'metric': 0.9166666666666666}]
```

Model: 120B
| Reasoning Effort | GPQA | AIME25 |
|------------------|------|--------|
| Low              | 65.0 |   52.5 |
| Medium           | 70.3 |   74.2 |
| High             | 79.3 |   91.7 |

## Collect environment

```bash
docker run -it --gpus all \
    -p 8000:8000 \
    --ipc=host \
    -v $HF_HOME:/root/.cache/huggingface \
    --entrypoint "" \
    vllm/vllm-openai:gptoss \
    vllm collect-env
```
