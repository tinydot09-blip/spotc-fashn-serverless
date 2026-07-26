FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install --ignore-installed runpod gradio

CMD ["python", "-u", "handler.py"]
