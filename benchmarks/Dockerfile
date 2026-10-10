FROM nvcr.io/nvidia/pytorch:24.10-py3

# The base image provides the torch==2.5.1+cu124 build used in the report.
# The frozen dependency snapshot of the reported experiment is installed from
# benchmarks/requirements-repro.txt (see also benchmarks/README.md).

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r benchmarks/requirements-repro.txt

ENV HF_HOME=/workspace/.hf-cache

# Example (requires an NVIDIA GPU):
#   docker build -f benchmarks/Dockerfile -t hakuli-bench .
#   docker run --rm --gpus all -v $(pwd)/benchmarks/results:/workspace/benchmarks/results \
#       hakuli-bench python benchmarks/run_benchmark.py \
#       --config benchmarks/configs/memory_llama7b_hakuli.json
