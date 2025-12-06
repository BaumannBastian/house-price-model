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

# Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Restlichen Code ins Image kopieren
COPY . .

# Standardkommando: Training starten
CMD ["python", "train.py"]