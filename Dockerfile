# docker build -t ldvqn .
# docker run --gpus all --shm-size=16g -it --rm \
#   -v /path/to/data:/workspace/LD-VQN-Reid/data \
#   -v /path/to/weights:/workspace/LD-VQN-Reid/weights \
#   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#   ldvqn bash scripts/eval_veri776.sh
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/LD-VQN-Reid
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["bash"]
