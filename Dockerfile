# ------------------------------
# Dockerfile
#
# Dieses Dockerfile baut ein Docker-Image für das House-Price-Projekt
# (Training und Prediction) inkl. aller benötigten Python-Abhängigkeiten.
# ------------------------------

# Basis: schlankes Python unter Linux
FROM python:3.11-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Dependencies installieren
COPY requirements.txt .
RUN python -m pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Restlichen Code ins Image kopieren
COPY . .

# Standardkommando: Training starten
CMD ["python", "train.py"]