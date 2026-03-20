 I exported my currently available Conda dependency configuration to this file.

### Core Runtime Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| **vllm** | 0.17.0 | High-throughput LLM inference engine |
| **torch** | 2.10.0 | PyTorch deep learning framework |
| **transformers** | 5.3.0 | Hugging Face transformers library |
| **fastapi** | 0.135.1 | Web framework for API server |
| **uvicorn** | 0.41.0 | ASGI server for FastAPI |
| **anthropic** | 0.84.0 | Anthropic SDK for proxy |
| **openai** | 2.24.0 | OpenAI SDK |

### CUDA/NVIDIA Stack

| Package | Version | Description |
|---------|---------|-------------|
| **cuda-toolkit** | 12.6.0 | NVIDIA CUDA Toolkit |
| **cuda-python** | 12.9.4 | CUDA Python bindings |
| **nvidia-cublas-cu12** | 12.8.4.1 | cuBLAS library |
| **nvidia-cudnn-cu12** | 9.10.2.21 | cuDNN library |
| **nvidia-nccl-cu12** | 2.27.5 | NCCL communication library |
| **nvidia-cusolver-cu12** | 11.7.3.90 | cuSOLVER library |
| **nvidia-cusparse-cu12** | 12.5.8.93 | cuSPARSE library |
| **nvidia-cufft-cu12** | 11.3.3.83 | cuFFT library |
| **cupy-cuda12x** | 14.0.1 | CuPy for GPU arrays |

### ML/Framework Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| **torchaudio** | 2.10.0 | PyTorch audio library |
| **torchvision** | 0.25.0 | PyTorch vision library |
| **triton** | 3.6.0 | Triton programming language |
| **xformers** | 0.0.29.post2 | Memory-efficient transformers |
| **accelerate** | 1.13.0 | Hugging Face accelerator |
| **safetensors** | 0.7.0 | Safe tensor format |
| **flashinfer-python** | 0.6.4 | Flash attention inference |

### Supporting Libraries

| Package | Version |
|---------|----------|
| **numpy** | 2.4.3 |
| **scipy** | 1.17.1 |
| **pillow** | 12.1.1 |
| **pyyaml** | 6.0.3 |
| **requests** | 2.32.5 |
| **httpx** | 0.28.1 |
| **pydantic** | 2.12.5 |
| **jinja2** | 3.1.6 |
| **tiktoken** | 0.12.0 |
| **tokenizers** | 0.22.2 |
| **sentencepiece** | 0.2.1 |

### System/Infrastructure

| Package | Version |
|---------|----------|
| **python** | 3.11.15 |
| **pip** | 26.0.1 |
| **setuptools** | 82.0.1 |
| **wheel** | 0.46.3 |
| **ray** | 2.54.0 |
| **psutil** | 7.2.2 |
| **loguru** | 0.7.3 |
| **tqdm** | 4.67.3 |

---