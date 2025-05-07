FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu
WORKDIR /app
RUN apt update && \
    apt install -y --no-install-recommends libportaudiocpp0 libportaudio2 && \
    pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    sounddevice \
    fastapi \
    uvicorn \
    python-multipart
COPY speech_recognition.py .
COPY server.py .
ENV HF_HOME="/huggingface/"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]