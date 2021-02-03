FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update --fix-missing && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pytorch-lightning===1.1.2 numpy jupyter pillow guppy3 pytorch_warmup && \
    rm -rf /var/lib/apt/lists/*

COPY img_gen_config/ data/
COPY bin/ bin/
COPY src/OCR/ OCR/
COPY src/LPR/ LPR/

ENV PYTHONPATH="$PYTHONPATH:/OCR:/LPR"
