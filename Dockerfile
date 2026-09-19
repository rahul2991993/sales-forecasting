FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts

RUN useradd --create-home appuser
USER appuser

CMD ["fastapi", "run", "src/api.py", "--host", "0.0.0.0", "--port", "8080"]