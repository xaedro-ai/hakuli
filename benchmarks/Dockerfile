FROM python:3.11-slim

# Reproduces the software environment of the reported experiment:
# Python 3.11, PyTorch 2.5.1+cu124, Transformers 4.57.6, Datasets 5.0.1.
# benchmarks/requirements-repro.txt pins the historical versions and points
# pip at the official PyTorch CUDA 12.4 wheel index for the +cu124 build.

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r benchmarks/requirements-repro.txt

ENV HF_HOME=/workspace/.hf-cache

# Example (requires an NVIDIA GPU):
#   docker build -f benchmarks/Dockerfile -t hakuli-bench .
#   docker run --rm --gpus all -v $(pwd)/benchmarks/results:/workspace/benchmarks/results \
#       hakuli-bench python benchmarks/run_benchmark.py \
#       --config benchmarks/configs/memory_llama7b_hakuli.json
