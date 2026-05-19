FROM unsloth/unsloth:latest

WORKDIR /workspace

RUN pip install datasets

COPY train.py /workspace/train.py

RUN mkdir -p /workspace/outputs /workspace/data

ENTRYPOINT ["python", "/workspace/train.py"]