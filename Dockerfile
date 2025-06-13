FROM nvcr.io/nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DGGML_CUDA=on"

RUN pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/llmsrv.py .

EXPOSE 8080

CMD ["uvicorn", "llmsrv:app", "--port", "8080"]
