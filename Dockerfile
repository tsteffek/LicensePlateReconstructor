FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get update --fix-missing && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pytorch-lightning===1.1.7 numpy jupyter pillow guppy3 pytorch_warmup && \
    rm -rf /var/lib/apt/lists/*

COPY img_gen_config/ img_gen_config/
COPY bin/ bin/
COPY src/ src/

ENV PYTHONPATH="$PYTHONPATH:/src"
