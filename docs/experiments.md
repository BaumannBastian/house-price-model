# Experimente & Modellvergleich

Hier dokumentiere ich wichtige Modell-Experimente (Metriken, Erkenntnisse).

## Aktueller Stand (Kurzfassung)

- Beste CV-Performance: HistGradientBoosting mit log-Target (HistGBR_log)
- Typischer Fehler:
  - RMSE ≈ 29–30k €
  - MARE ≈ 9–10 %
- TorchMLP:
  - R² ~ 0.87
  - höhere relative Fehler als HistGBR_log