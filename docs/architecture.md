# Architekturübersicht

Dieses Projekt ist ein End-to-End-ML-System zur Vorhersage von Hauspreisen.

## Lokale Architektur

- `src/features.py`: Missing-Value-Treatment, Feature Engineering, Ordinal-Encoding
- `src/preprocessor.py`: ColumnTransformer (One-Hot + numerische Features)
- `src/models.py`: klassische Modelle (LinearRegression, RandomForest, HistGBR)
- `src/nn_models.py`: TorchMLPRegressor (PyTorch-MLP)
- `train.py`: trainiert alle Modelle, vergleicht Metriken, speichert Champion
- `predict.py`: lädt Champion und erstellt Vorhersagen als CSV

## Ziel-Cloud-Architektur (Entwurf)

- S3-Bucket:
  - Speicherung von Trainingsdaten (CSV) und gelernten Modellen (`.joblib`)
- Postgres (z.B. AWS RDS):
  - Tabelle(n) für Vorhersagen und ggf. Rohdaten
- Compute (z.B. AWS ECS Fargate mit Docker-Image):
  - Service, der das Modell lädt und Vorhersagen erzeugt
- (Später) API-Gateway oder Load Balancer:
  - HTTP-Schnittstelle für Predictions