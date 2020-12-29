FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update --fix-missing && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pytorch-lightning numpy jupyter pillow && \
    rm -rf /var/lib/apt/lists/*

COPY data/ data/
COPY bin/ bin/
COPY src/ src/

ENV PYTHONPATH="$PYTHONPATH:/src"