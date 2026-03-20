#!/bin/bash
# 启动 Anthropic API 代理 → 连接 vLLM 后端
# 用于将 Claude Code 的 Anthropic 协议请求翻译为 OpenAI 协议

CONDA_ENV=/path-to-conda-env/qwen35_opus
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VLLM_URL=${VLLM_URL:-http://localhost:8766}
PROXY_PORT=${PROXY_PORT:-8800}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-27B-Opus}

source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

echo "=== Anthropic API Proxy ==="
echo "Proxy:  0.0.0.0:${PROXY_PORT}"
echo "vLLM:   ${VLLM_URL}"
echo "Model:  ${MODEL_NAME}"
echo "==========================="

python "${SCRIPT_DIR}/anthropic_proxy.py" \
  --vllm-url "${VLLM_URL}" \
  --port "${PROXY_PORT}" \
  --model-name "${MODEL_NAME}"
