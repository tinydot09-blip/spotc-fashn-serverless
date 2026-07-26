FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install --ignore-installed runpod gradio

RUN python scripts/download_weights.py --weights-dir /app/weights

ENV FASHN_WEIGHTS_DIR=/app/weights

CMD ["python", "-u", "handler.py"]
