# House Price Model

End-to-End-ML-Projekt zur Vorhersage von Hauspreisen (Ames Housing Dataset).
Ziel: saubere Pipeline wie in Data-Science-/Data-Engineering-Interviews:
Preprocessing → Feature Engineering → Modelle → Evaluation → Deployment (Docker, später AWS / Postgres / Tableau).

## Projektstruktur

- `src/`
  - `features.py` – Missing-Value-Treatment, Feature Engineering, Ordinal-Encoding
  - `preprocessor.py` – ColumnTransformer (One-Hot-Encoding + numerische Features)
  - `models.py` – klassische Modelle (LinearRegression, RandomForest, HistGBR)
  - `nn_models.py` – PyTorch-MLP (TorchMLPRegressor)
- `train.py` – trainiert alle Modelle, vergleicht Metriken, speichert Champion (`models/*.joblib`)
- `predict.py` – lädt Champion und macht Vorhersagen (z.B. auf `data/raw/test.csv`)
- `Dockerfile` – Build für die Trainings-Pipeline im Container
- `data/` – raw / processed Daten (nicht im Repo)
- `models/`, `logs/`, `plots/`, `predictions/` – Artefakte (nicht im Repo)

## Daten

- Basis: Kaggle "House Prices: Advanced Regression Techniques"
- Train-Datei: `data/raw/train.csv` (lokal, nicht versioniert)

## Pipeline (kurz)

1. Laden & Preprocessing  
   - Missing Values: None/Median/Modus-Strategien (siehe `features.py`)
   - Feature Engineering: `TotalSF`, `TotalBath`, `HouseAge`, etc.
   - Ordinale Features: Mapping auf `int` (z.B. Ex/Gd/TA/Fa/Po → 5…1)

2. Modelle
   - LinearRegression (+ optional log-Target via `TransformedTargetRegressor`)
   - RandomForestRegressor
   - HistGradientBoostingRegressor
   - TorchMLPRegressor (PyTorch-Multi-Layer-Perceptron)

3. Evaluation
   - Train/Test-Split (80/20)
   - Cross-Validation (5-fold)
   - Metriken:
     - R²
     - RMSE (in Euro)
     - MARE / RRMSE (relative Fehler in %)

## Aktueller Stand (kurze Summary)

- Bester CV-Performer: **HistGBR mit log-Target** (`HistGBR_log`)
- Typischer Fehler:
  - ca. 9–10 % relativer Fehler (MARE),
  - RMSE ~ 29–30k € (je nach Modell)
- TorchMLP:
  - R² ~ 0.87,
  - aber schlechtere relative Fehler und höhere CV-RMSE als HistGBR_log.

## Wie ausführen

### Lokal (Python-Umgebung)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

python train.py   # trainiert alle Modelle, speichert Champion
python predict.py # erstellt predictions/predictions.csv mit Champion
