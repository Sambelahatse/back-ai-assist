FROM python:3.11-slim

WORKDIR /app

# Variables Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Code
COPY . .

# Port Hugging Face
EXPOSE 7860

# Gunicorn (HF-compatible)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "index:app"]
