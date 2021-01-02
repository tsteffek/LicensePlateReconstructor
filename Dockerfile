FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update --fix-missing && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pytorch-lightning===1.1.2 numpy jupyter pillow guppy3 && \
    rm -rf /var/lib/apt/lists/*

COPY data/ data/
COPY bin/ bin/
COPY OCR/ OCR/
COPY LPR/ LPR/

ENV PYTHONPATH="$PYTHONPATH:/OCR:/LPR"
