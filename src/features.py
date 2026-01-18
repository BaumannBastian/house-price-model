# ------------------------------------
# src/features.py
#
# In dieser Python-Datei werden Funktionen für Missing-Value-Treatment,
# Feature-Engineering und ordinal Encoding des House-Prices-Datensatzes
# bereitgestellt.
#
# Wichtiger Hinweis (Leakage vermeiden)
# ------------------------------------
# Alles, was statistische Kennzahlen aus den Daten braucht (Median/Modus),
# wird in einer sklearn-Pipeline pro Fold gefittet. (Siehe src/preprocessing.py)
# Diese Datei macht deshalb nur:
# - domain-sinnvolle, konstante Fills (z.B. "None" für "nicht vorhanden")
# - rein zeilenweises Feature-Engineering (keine globalen Aggregationen)
# - ordinal Mapping (deterministisch)
# ------------------------------------

from __future__ import annotations

import pandas as pd


# Missing-Value Treatment
def missing_value_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Behandelt Missing Values ohne Statistik (kein Leakage).

    Strategie:
    - Bestimmte kategoriale Spalten werden mit dem String ``"None"`` gefüllt,
      um "nicht vorhanden" explizit zu codieren (z.B. kein Pool, keine Garage).
    - Ausgewählte numerische Spalten, bei denen 0 semantisch "nicht vorhanden"
      bedeutet, werden mit ``0`` gefüllt (z.B. GarageYrBlt falls keine Garage).

    Alles andere (Median/Modus) macht anschließend der Preprocessor in
    ``src/preprocessing.py`` via ``SimpleImputer``.

    Parameter
    ----------
    df : pandas.DataFrame
        Roh-DataFrame mit möglichen fehlenden Werten.

    Returns
    -------
    pandas.DataFrame
        Kopie von ``df`` mit konstant imputierten Spalten.
    """
    df = df.copy()

    none_fill = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageQual",
        "GarageFinish",
        "GarageType",
        "GarageCond",
        "BsmtExposure",
        "BsmtCond",
        "BsmtQual",
        "BsmtFinType2",
        "BsmtFinType1",
    ]
    zero_fill = [
        "GarageYrBlt",
    ]

    for col in none_fill:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# Feature Engineering
def new_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Erstellt neue, zusammengesetzte Features im DataFrame (rein zeilenweise).

    Features (Auszug):
    - ``HouseAge``      : Alter des Hauses beim Verkauf (YrSold - YearBuilt)
    - ``RemodAge``      : Alter seit Renovierung (YrSold - YearRemodAdd)
    - ``IsRemodeled``   : 1 falls renoviert, sonst 0
    - ``TotalSF``       : TotalBsmtSF + 1stFlrSF + 2ndFlrSF
    - ``TotalBath``     : FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
    - ``TotalPorchSF``  : OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF
    - ``HasBasement``   : 1 falls TotalBsmtSF > 0
    - ``HasGarage``     : 1 falls GarageArea > 0
    - ``HasFireplace``  : 1 falls Fireplaces > 0
    - ``HasPool``       : 1 falls PoolArea > 0
    - ``QualSF``        : OverallQual * GrLivArea

    Hinweis: Für Summen/Produkte nutzen wir ``fillna(0)``, damit fehlende
    Komponenten nicht zu NaNs in den neuen Features führen. Die finale
    (statistische) Imputation passiert dennoch im Preprocessor.

    Parameter
    ----------
    df : pandas.DataFrame
        DataFrame mit den Original-Spalten des House-Prices-Datensatzes.

    Returns
    -------
    pandas.DataFrame
        Derselbe DataFrame (in-place erweitert).
    """
    # Altersfeatures
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    if "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # TotalSF
    required_sf = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
    for col in required_sf:
        if col not in df.columns:
            raise KeyError(f"Spalte fehlt für Feature-Engineering: {col}")
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    # TotalBath
    required_bath = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    for col in required_bath:
        if col not in df.columns:
            raise KeyError(f"Spalte fehlt für Feature-Engineering: {col}")
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    # TotalPorchSF
    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    for col in porch_cols:
        if col not in df.columns:
            raise KeyError(f"Spalte fehlt für Feature-Engineering: {col}")
    df["TotalPorchSF"] = sum(df[c].fillna(0) for c in porch_cols)

    # Binäre Has_* Features
    if "TotalBsmtSF" in df.columns:
        df["HasBasement"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)

    # Interaktion Qualität x Fläche
    if "OverallQual" in df.columns and "GrLivArea" in df.columns:
        df["QualSF"] = df["OverallQual"].fillna(0) * df["GrLivArea"].fillna(0)

    return df


# Ordinal Encoding
def ordinal_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Kodiert ordinale kategoriale Merkmale in geordnete numerische Werte.

    Mapping-Dictionaries übersetzen typische Qualitäts- und Funktionsstufen
    (z.B. ``Ex``, ``Gd``, ``TA``) in ganze Zahlen.

    Wichtig:
    - Vorher sollte ``missing_value_treatment`` gelaufen sein, damit echte
      "nicht vorhanden"-Fälle als ``"None"`` vorliegen.
    - Echte NaNs können trotzdem vorkommen; die verbleibende Imputation
      macht danach der Preprocessor (SimpleImputer).

    Parameter
    ----------
    df : pandas.DataFrame
        DataFrame nach (konstantem) Missing-Value-Treatment.

    Returns
    -------
    pandas.DataFrame
        Derselbe DataFrame, ordinale Spalten sind numerisch kodiert.
    """
    quality_mapping_ExGdTAFaPoNone = {
        "Ex": 5,  # Excellent
        "Gd": 4,  # Good
        "TA": 3,  # Typical/Average
        "Fa": 2,  # Fair
        "Po": 1,  # Poor
        "None": 0,  # None
    }
    quality_mapping_GdAvMnNoNone = {
        "Gd": 4,  # Good Exposure
        "Av": 3,  # Average Exposure
        "Mn": 2,  # Minimum Exposure
        "No": 1,  # No Exposure
        "None": 0,  # None
    }
    quality_mapping_GLQALQBLQRecLwQUnfNone = {
        "GLQ": 6,  # Good Living Quarters
        "ALQ": 5,  # Average Living Quarters
        "BLQ": 4,  # Below Average Living Quarters
        "Rec": 3,  # Average Rec Room
        "LwQ": 2,  # Low Quality
        "Unf": 1,  # Unfinished
        "None": 0,  # None
    }
    quality_mapping_TypMin1Min2ModMaj1Maj2SevSal = {
        "Typ": 7,  # Typical Functionality
        "Min1": 6,  # Minor Deductions 1
        "Min2": 5,  # Minor Deductions 2
        "Mod": 4,  # Moderate Deductions
        "Maj1": 3,  # Major Deductions 1
        "Maj2": 2,  # Major Deductions 2
        "Sev": 1,  # Severely Damaged
        "Sal": 0,  # Salvage Only
    }
    quality_mapping_FinRFnUnfNone = {
        "Fin": 3,  # Finished
        "RFn": 2,  # Rough Finished
        "Unf": 1,  # Unfinished
        "None": 0,  # None
    }
    quality_mapping_YPN = {
        "Y": 2,  # Paved
        "P": 1,  # Partially Paved
        "N": 0,  # Dirt/Gravel
    }

    cols_ExGdTAFaPoNone = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    ]
    cols_GdAvMnNoNone = ["BsmtExposure"]
    cols_GLQALQBLQRecLwQUnfNone = ["BsmtFinType1", "BsmtFinType2"]
    cols_TypMin1Min2ModMaj1Maj2SevSal = ["Functional"]
    cols_FinRFnUnfNone = ["GarageFinish"]
    cols_YPN = ["PavedDrive"]

    for col in cols_ExGdTAFaPoNone:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping_ExGdTAFaPoNone)

    for col in cols_GdAvMnNoNone:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping_GdAvMnNoNone)

    for col in cols_GLQALQBLQRecLwQUnfNone:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping_GLQALQBLQRecLwQUnfNone)

    for col in cols_TypMin1Min2ModMaj1Maj2SevSal:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping_TypMin1Min2ModMaj1Maj2SevSal)

    for col in cols_FinRFnUnfNone:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping_FinRFnUnfNone)

    for col in cols_YPN:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping_YPN)

    return df
