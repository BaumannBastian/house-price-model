# Experiments – House Price Model

Dieses Dokument sammelt Experimente, Modelle und Beobachtungen (ML-seitig). Es ist als Logbook gedacht, nicht als “How-To”.

---

## 1) Datensatz

- Kaggle: House Prices – Advanced Regression Techniques
- Zielvariable: `SalePrice`
- Eigenschaften:
  - rechtsschief verteilt → Log-Transformation ist oft stabiler
  - gemischte Features: numerisch + kategorisch + Missing Values

---

## 2) Target-Transformation

Zwei Varianten:

- ohne Log-Target
  - Regression direkt auf `SalePrice`
- mit Log-Target
  - Regression auf `log1p(SalePrice)`
  - Rücktransformation via `expm1` (z.B. `TransformedTargetRegressor`)

Beobachtung:
- Log-Target stabilisiert häufig Fehler und reduziert Fold-Varianz, besonders bei linearen Modellen.
- Bei Baumverfahren ist der Effekt oft kleiner, kann aber dennoch helfen.

---

## 3) Ergebnisse (Stand: zuletzt gemessener Run)

KFold: 5, Seed: 42  
Metriken: CV-RMSE (Mittel ± Std), zusätzlich Holdout/Test-Split RMSE und R² aus dem Trainingslauf.

| Modell               | Log-Target | CV-RMSE (±)            | Test RMSE  | Test R²  | Kommentar |
|---------------------|-----------:|------------------------:|-----------:|---------:|----------|
| HistGBR_log         | ja         | 25419.61 ± 3384.36      | 28926.15   | 0.8909   | stabilster CV → Champion |
| HistGBR             | nein       | 27021.45 ± 3182.30      | 29901.99   | 0.8834   | starke Tabular-Baseline |
| RandomForest_log    | ja         | 29189.22 ± 4910.20      | 29538.61   | 0.8862   | solide, etwas mehr Varianz |
| RandomForest        | nein       | 29257.03 ± 4108.57      | 29716.36   | 0.8849   | ähnlich zu RF_log |
| TorchMLP            | nein       | 33233.23 ± 5817.77      | 27162.81   | 0.9038   | guter Holdout, schwächer/stärker variabel im CV |
| LinearRegression    | nein       | 35435.39 ± 7186.45      | 30480.23   | 0.8789   | einfache Baseline |
| LinearRegression_log| ja         | 35986.32 ± 13963.16     | 28611.27   | 0.8933   | teils hohe Fold-Varianz |
| TorchMLP_log        | ja         | 41142.95 ± 14042.89     | 31025.72   | 0.8745   | aktuell instabil |

Interpretation:
- Tabular-Boosting (HistGBR) liefert die beste Kombination aus Performance und Stabilität.
- TorchMLP zeigt Potential (Holdout gut), aber CV-Stabilität ist der Engpass.

---

## 4) TorchMLP – Beobachtungen (kurz)

- Optimizer: Adam
- Loss: MSE
- Early Stopping mit Validation-Split (intern)
- Typische Failure-Modes:
  - hohe Fold-Varianz (sensitiv auf Split/Seed)
  - Overfitting bei zu wenig Regularisierung
  - Inkonsistenzen bei Target-Transforms (Log/No-Log) im Training vs. Inference

---

## 5) Nächste Experiment-Ideen

- Feature Engineering: Interaktionen, robuste Outlier-Features, seltene Kategorien konsolidieren
- Hyperparameter-Suche (kleiner, sinnvoller Suchraum)
- TorchMLP: Regularisierung, Dropout, Weight Decay, LR-Schedules, stabilere Normalisierung