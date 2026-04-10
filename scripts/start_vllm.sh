#!/bin/bash
###
 # @Author: zhouyuchong
 # @Date: 2026-04-09 11:45:09
 # @Description: 
 # @LastEditors: zhouyuchong
 # @LastEditTime: 2026-04-09 13:29:15
### 
# Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled  vLLM 服务
# 提供 OpenAI 兼容 API，配合 anthropic_proxy.py 接入 Claude Code

CONDA_ENV=/usr/local
MODEL=/root/workspace/models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2
NVIDIA_LIBS=${CONDA_ENV}/lib/python3.11/site-packages

export PATH=${CONDA_ENV}/bin:$PATH
export CUDA_HOME=${CONDA_ENV}

export LIBRARY_PATH=\
${CONDA_ENV}/targets/x86_64-linux/lib:\
${NVIDIA_LIBS}/nvidia/cublas/lib:\
${NVIDIA_LIBS}/nvidia/cuda_runtime/lib:\
/usr/local/cuda/lib64

export LD_LIBRARY_PATH=\
${NVIDIA_LIBS}/nvidia/cublas/lib:\
${NVIDIA_LIBS}/nvidia/cuda_runtime/lib:\
${NVIDIA_LIBS}/nvidia/nccl/lib:\
${NVIDIA_LIBS}/nvidia/cudnn/lib:\
${NVIDIA_LIBS}/nvidia/cufft/lib:\
${NVIDIA_LIBS}/nvidia/curand/lib:\
${NVIDIA_LIBS}/nvidia/cusolver/lib:\
${NVIDIA_LIBS}/nvidia/cusparse/lib:\
${CONDA_ENV}/lib:\
/usr/local/cuda/lib64

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}

PORT=${PORT:-8766}
TP=${TP:-1}
MAX_LEN=${MAX_LEN:-131072}

echo "=== Qwen3.5-4B-Opus vLLM Server ==="
echo "Port: ${PORT}  TP: ${TP}  MaxLen: ${MAX_LEN}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "======================================"

${CONDA_ENV}/bin/vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size ${TP} \
  --max-model-len ${MAX_LEN} \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen3.5-4B-Opus \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
  --quantization fp8