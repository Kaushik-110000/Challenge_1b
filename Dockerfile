FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    DGLBACKEND=pytorch

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
      libmagic-dev poppler-utils git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
      torch==2.0.0+cpu \
      torchaudio==2.0.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY process_all_pdfs.py process_pdfs.py best_model.pth ./

CMD ["python", "process_all_pdfs.py"]
