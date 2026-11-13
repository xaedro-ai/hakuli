FROM python:3.11-slim

# Maintained environment for new benchmark runs.
# Historical measurement versions are documented in historical-environment.md.

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r benchmarks/requirements.txt \
    && pip check

ENV HF_HOME=/workspace/.hf-cache

# Example (requires an NVIDIA GPU):
#   docker build -f benchmarks/Dockerfile -t hakuli-bench .
#   docker run --rm --gpus all -v $(pwd)/benchmarks/results:/workspace/benchmarks/results \
#       hakuli-bench python benchmarks/run_benchmark.py \
#       --config benchmarks/configs/memory_llama7b_hakuli.json
