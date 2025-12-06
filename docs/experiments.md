# Experiments – House Price Model

Dieses Dokument sammelt wichtige Experimente, Modelle, Hyperparameter und Beobachtungen.

---

## 1. Datensatz

- Kaggle-Competition: **House Prices – Advanced Regression Techniques**
- Zielvariable:
  - `SalePrice` (Verkaufspreis in Dollar)
- Beobachtungen:
  - Zielwert ist stark rechtsschief → Log-Transformation sinnvoll
  - Feature-Spektrum: Mischung aus
    - kontinuierlichen Größen (Flächen, Jahre, etc.)
    - kategorischen Features (Qualität, Material, Lage, …)
    - ordinalen Qualitätsmerkmalen

---

## 2. Modelle

### 2.1 Basismodelle (sklearn)

1. **LinearRegression**
   - ohne Regularisierung
   - arbeitet auf den One-Hot-encodierten + numerischen Features

2. **RandomForestRegressor**
   - Ensembles von Entscheidungsbäumen
   - robust gegenüber Ausreißern und Feature-Skalierung
   - Hyperparameter grob gesetzt (z.B. Anzahl Bäume, Tiefe)

3. **HistGradientBoostingRegressor**
   - Gradient Boosting mit Histogram-basierten Splits
   - unterstützt Missing Values intern recht gut
   - in vielen Tabular-Setups sehr stark

### 2.2 Neuronales Netz (PyTorch)

4. **TorchMLPRegressor**
   - selbst definierter sklearn-kompatibler Wrapper um ein PyTorch-MLP
   - Architektur (typisches Setup):
     - Input-Layer: Dimension = Anzahl Features nach Preprocessing
     - Hidden-Layer: z.B. `(128, 64)` Neuronen mit ReLU
     - Output-Layer: 1 Neuron (Regression)
   - Optimierung:
     - Loss: MSE
     - Optimizer: Adam (z.B. `lr=1e-3` bis `1e-2`)
     - Batch Size: z.B. 64–128
     - Epochen: bis 1000, mit Early Stopping
     - Learning-Rate-Scheduler: `StepLR` (LR-Decay während des Trainings)
   - Features werden vor dem Netz mit `StandardScaler` skaliert (innerhalb der Pipeline).

### 2.3 Log-Transformation des Targets

Für alle klassischen Modelle existieren zwei Varianten:

- **ohne Log-Target**  
  → Direkte Regression auf `SalePrice` in Dollar.

- **mit Log-Target**  
  → Regression auf `log1p(SalePrice)`  
  → Rücktransformation mit `expm1` (via `TransformedTargetRegressor`)

Motivation:

- stabilere Fehlermaßverteilung
- relative Fehler werden stärker gewichtet
- linearere Zusammenhänge für lineare Modelle

---

## 3. Ergebnisse (Stand: aktueller Projektstatus)

Die genauen Werte können leicht schwanken (je nach Seed / Split).  
Angegeben sind typische Größenordnungen.

| Modell                  | Log-Target | Test R² | Test RMSE  | MARE   | RRMSE  | CV-RMSE (ca.) | Kommentar                            |
|-------------------------|-----------:|--------:|-----------:|-------:|-------:|--------------:|--------------------------------------|
| LinearRegression         | nein      | ~0.88   | ~30.7k €   | ~12 %  | ~18 % | ~38k €        | einfache Baseline                    |
| **LinearRegression_log** | **ja**   | **~0.93** | **~22.8k €** | **~9 %** | **~14 %** | ~49k € (sehr instabil) | bestes RMSE auf Test, aber CV volatil |
| RandomForest             | nein      | ~0.89   | ~29.2k €   | ~10.5% | ~18 % | ~29.5k €      | solide, leicht besser als Linear     |
| RandomForest_log         | ja        | ~0.88   | ~30.4k €   | ~10 %  | ~17 % | ~30.0k €      | Log-Target bringt wenig              |
| HistGradientBoosting     | nein      | ~0.89   | ~29.0k €   | ~9.8 % | ~17 % | ~28.5k €      | sehr robust                          |
| **HistGBR_log**          | **ja**   | ~0.88   | ~30.1k €   | **~9.5 %** | **~16.7 %** | **~27.3k €** | **bester CV-RMSE → Champion**   |
| TorchMLP                 | nein      | ~0.87–0.88 | ~31–33k € | ~14 % | >20 % | >40k €        | trotz Tuning aktuell schwächer       |

**Interpretation:**

- **LinearRegression_log** hat auf dem konkreten Test-Split das beste RMSE,  
  aber die Cross-Validation zeigt eine hohe Varianz → overfitting-/Instabilitätsgefahr.
- **HistGBR_log** ist über alle Folds hinweg deutlich stabiler und liefert den besten CV-RMSE → wird als Champion-Modell gewählt.
- Der Vorteil von Log-Target ist für lineare Modelle sehr deutlich, für Baumverfahren moderat.
- Das MLP braucht deutlich mehr Daten/Tuning, um an die Boosted Trees heranzukommen; mit dem aktuellen Setup bleibt es etwas zurück.

---

## 4. TorchMLP – Detaillierte Beobachtungen

### 4.1 Trainings-Setup

- Features:
  - Ausgangspunkt: Output des `ColumnTransformer` + `StandardScaler`
- Optimierung:
  - MSELoss
  - Adam (z.B. `lr = 1e-3` oder `1e-2`)
  - Batch-Größen: 64–128 getestet
  - `max_epochs`: bis zu 1000
  - Early Stopping:
    - Validierungsanteil `val_fraction` (z.B. 10 % von X)
    - `patience` (z.B. 20 Epochen)
    - `min_delta` für „echte“ Verbesserungen
  - Learning-Rate-Scheduler:
    - `StepLR` mit schrittweiser Reduktion der Lernrate

### 4.2 Training-Verlauf

- Loss-Kurven (Train/Val) zeigen:
  - abnehmenden Train-Loss
  - gelegentliches Plateau beim Val-Loss → Early Stopping greift
- Trotz Optimierungsschritten:
  - Test-RMSE bleibt über dem von HistGBR
  - relative Fehler (MARE, RRMSE) sind höher
  - CV-RMSE ist schlechter

### 4.3 Ursachen / Hypothesen

- Datensatzgröße (~1460 Trainingspunkte) ist für ein MLP relativ klein
- Bäume (insbesondere Gradient Boosting) sind im Tabular-Setting oft im Vorteil
- Potenzial für weitere Verbesserungen:
  - Regularisierung (Dropout, Weight Decay)
  - andere Architektur (weniger Tiefe/Breite → weniger Overfitting)
  - andere Aktivierungsfunktionen (z.B. SiLU, GELU)
  - bessere Initialisierung / Lernraten-Planung

---

## 5. Feature Engineering – bisher und Ideen

### 5.1 Bisher implementiert

- `TotalSF`: kombinierte Wohn-/Kellerfläche
- `TotalBath`: gewichtete Summe über Full/Half-Baths inkl. Basement
- `HouseAge`: `YrSold - YearBuilt`
- `RemodAge`: `YrSold - YearRemodAdd`
- `TotalPorchSF`: Summe verschiedener Porch-Features
- Ordinale Qualitätsmerkmale auf Integer-Skalen abgebildet

Diese Features haben die Performance insbesondere der linearen Modelle verbessert und auch den Baummodellen zusätzliche Struktur gegeben.

### 5.2 Ideen für weitere Features

- Interaktionen:
  - Qualität × Fläche (z.B. `OverallQual * TotalSF`)
  - Qualität × Alter
- Normalisierung/Skalierung bestimmter Extrema:
  - Log-Transformation einzelner extrem schief verteilter Features (z.B. `LotArea`)
- Binning:
  - Jahr-Bins (z.B. Baujahr-Dekaden)
  - Alters-Kategorien

---

## 6. Nächste geplante Experimente

1. **Gezieltes Hyperparameter-Tuning für HistGBR_log**
   - Parameter wie:
     - `learning_rate`
     - `max_depth`
     - `max_iter`
   - z.B. mit `RandomizedSearchCV` auf einem eingeschränkten, sinnvollen Suchraum.

2. **Systematisches Tuning für TorchMLP**
   - Suchraum:
     - `hidden_dims` (z.B. (64,), (128, 64), (256, 128, 64))
     - `lr` (1e-4 … 1e-2)
     - `batch_size` (32, 64, 128)
   - Evaluationskriterium: CV-RMSE, MARE

3. **Erweitertes Feature Engineering**
   - mehr Interaktionen + ggf. log-transformierte Eingabefeatures
   - Feature-Importance-Analysen (Permutation Importance für HistGBR) zur Feature-Auswahl

4. **Weitere Modellklassen**
   - XGBoost / LightGBM als Referenz für Baum-Boosting
   - evtl. Regularized Linear Models (Ridge/Lasso) mit log-Target

5. **Experiment-Tracking**
   - Nutzung der `models`-Tabelle als einfaches Experiment-Log
   - evtl. Erweiterung um:
     - Tagging (z.B. `experiment_name`)
     - Notizen pro Modell (z.B. „Feature-Set v2“, „mit neuen Interaktionen“)