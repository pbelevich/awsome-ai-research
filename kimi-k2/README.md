# Kimi-K2

## Resources
* [Technical Report](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)
* https://magazine.sebastianraschka.com/i/168650848/kimi

## Deployment

```bash
huggingface-cli download moonshotai/Kimi-K2-Instruct
export MODEL_NAME=moonshotai/Kimi-K2-Instruct
export MODEL_PATH=`python -c "from pathlib import Path; from huggingface_hub import hf_hub_download; print(Path(hf_hub_download('$MODEL_NAME', filename='config.json')).parent)"`
```

### vLLM

vLLM 0.9.2 does not support Kimi-K2

Install vLLM:
```bash
# pip install --upgrade pip
# pip install uv
# srun uv pip install vllm --torch-backend=auto
# pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/vllm-project/vllm.git
git checkout 61e20828da1639c05a7bb7d1592c4834e10b33b7 # before https://github.com/vllm-project/vllm/pull/20769 fix: https://github.com/vllm-project/vllm/pull/21020
cd vllm
pip install jinja2 # solves Marlin generation failed.  Result: "1"
pip install blobfile
srun pip install --no-cache-dir --upgrade --force-reinstall -v -e .
```

Launch vLLM server:
```bash
sbatch vllm/kimi-k2-vllm.sbatch
```

Test vLLM server:
```bash
TODO
```

### SGLang

Install SGLang:
```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.9.post2"
```

Launch SGLang server:
```bash
sbatch sgl/kimi-k2-sgl.sbatch
```

Test SGLang server:

```bash
curl http://${HOST}:30000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "moonshotai/Kimi-K2-Instruct",
      "object": "model",
      "created": 1752629925,
      "owned_by": "sglang",
      "root": "moonshotai/Kimi-K2-Instruct",
      "max_model_len": 131072
    }
  ]
}
```

```bash
curl -s http://${HOST}:30000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "moonshotai/Kimi-K2-Instruct",
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
  "id": "22a858d4a7904289bb99554e7049a2ef",
  "object": "chat.completion",
  "created": 1752630015,
  "model": "moonshotai/Kimi-K2-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "matched_stop": 163586
    }
  ],
  "usage": {
    "prompt_tokens": 23,
    "total_tokens": 31,
    "completion_tokens": 8,
    "prompt_tokens_details": null
  }
}
```
