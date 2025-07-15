# Kimi-K2

```bash
huggingface-cli download moonshotai/Kimi-K2-Instruct
```

## vLLM

vLLM 0.9.2 does not support Kimi-K2

```bash
# pip install --upgrade pip
# pip install uv
# srun uv pip install vllm --torch-backend=auto
# pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/vllm-project/vllm.git
cd vllm
srun pip install -v -e .
```

## SGLang

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.9.post2"
```