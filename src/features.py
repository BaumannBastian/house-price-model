# src/features.py

import pandas as pd


# Missing-Value Treatment
def missing_value_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Behandelt fehlende Werte im DataFrame.
    """

    df = df.copy()

    # Defining columns for different imputation strategies
    none_fill = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual', 'GarageFinish', 'GarageType', 'GarageCond',
        'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1'
    ]
    median_fill = [
        'LotFrontage', 'MasVnrArea'
    ]
    modus_fill = [
        'MasVnrType', 'Electrical'
    ]
    zero_fill = [
        'GarageYrBlt'
    ]

    # Imputation application
    for col in none_fill:
        df[col] = df[col].fillna("None")

    for col in median_fill:
        df[col] = df[col].fillna(df[col].median())

    for col in modus_fill:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in zero_fill:
        df[col] = df[col].fillna(0)

    return df

# Feature Engineering
def new_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt neue Features im DataFrame.
    """

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    
    return df

# Ordinal Encoding
def ordinal_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Übersetzt ordinale Kategorien in geordnete numerische Werte im DataFrame.
    """

    # Mapping Dictionaries
    quality_mapping_ExGdTAFaPoNone = {
        # ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC
        'Ex': 5,        # Excellent
        'Gd': 4,        # Good                             
        'TA': 3,        # Typical/Average
        'Fa': 2,        # Fair
        'Po': 1,        # Poor
        'None': 0       # None
    }
    quality_mapping_GdAvMnNoNone = {
        # BsmtExposure
        'Gd': 4,        # Good Exposure
        'Av': 3,        # Average Exposure
        'Mn': 2,        # Minimum Exposure
        'No': 1,        # No Exposure
        'None': 0       # None
    }
    quality_mapping_GLQALQBLQRecLwQUnfNone = {
        # BsmtFinType1, BsmtFinType2
        'GLQ': 6,      # Good Living Quarters
        'ALQ': 5,      # Average Living Quarters
        'BLQ': 4,      # Below Average Living Quarters
        'Rec': 3,      # Average Rec Room
        'LwQ': 2,      # Low Quality
        'Unf': 1,      # Unfinshed
        'None': 0      # None
    }
    quality_mapping_TypMin1Min2ModMaj1Maj2SevSal = {
        # Functional
        'Typ': 7,       # Typical Functionality
        'Min1': 6,      # Minor Deductions 1
        'Min2': 5,      # Minor Deductions 2
        'Mod': 4,       # Moderate Deductions
        'Maj1': 3,      # Major Deductions 1
        'Maj2': 2,      # Major Deductions 2
        'Sev': 1,       # Severely Damaged
        'Sal': 0        # Salvage Only
    }
    quality_mapping_FinRFnUnfNone = {
        # GarageFinish
        'Fin': 3,       # Finished
        'RFn': 2,       # Rough Finished
        'Unf': 1,       # Unfinished
        'None': 0       # None
    }
    quality_mapping_YPN = {
        # PavedDrive
        'Y': 2,        # Paved
        'P': 1,        # Partially Paved
        'N': 0         # Dirt/Gravel
    }

    # Mapping Columns to Apply
    cols_ExGdTAFaPoNone = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
        'FireplaceQu', 'GarageQual', 'GarageCond','PoolQC'
    ]
    cols_GdAvMnNoNone = [
        'BsmtExposure'
    ]
    cols_GLQALQBLQRecLwQUnfNone = [
        'BsmtFinType1', 'BsmtFinType2'
    ]
    cols_TypMin1Min2ModMaj1Maj2SevSal = [
        'Functional'
    ]
    cols_FinRFnUnfNone = [
        'GarageFinish'
    ]
    cols_YPN = [
        'PavedDrive'
    ]

    # Mapping application
    for col in cols_ExGdTAFaPoNone:
        df[col] = df[col].map(quality_mapping_ExGdTAFaPoNone)

    for col in cols_GdAvMnNoNone:
        df[col] = df[col].map(quality_mapping_GdAvMnNoNone)

    for col in cols_GLQALQBLQRecLwQUnfNone:
        df[col] = df[col].map(quality_mapping_GLQALQBLQRecLwQUnfNone)

    for col in cols_TypMin1Min2ModMaj1Maj2SevSal:
        df[col] = df[col].map(quality_mapping_TypMin1Min2ModMaj1Maj2SevSal)

    for col in cols_FinRFnUnfNone:
        df[col] = df[col].map(quality_mapping_FinRFnUnfNone)

    for col in cols_YPN:
        df[col] = df[col].map(quality_mapping_YPN)
    
    return df
