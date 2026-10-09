FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir \
    galore-torch q-galore-torch apollo-torch matplotlib

ENV HF_HOME=/workspace/.hf-cache

# Example (requires an NVIDIA GPU):
#   docker build -f benchmarks/Dockerfile -t hakuli-bench .
#   docker run --rm --gpus all -v $(pwd)/benchmarks/results:/workspace/benchmarks/results \
#       hakuli-bench python benchmarks/run_benchmark.py \
#       --config benchmarks/configs/memory_llama7b_hakuli.json
