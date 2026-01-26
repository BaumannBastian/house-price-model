# ------------------------------------
# cloud/databricks/03_gold_features.py
#
# Gold-Features: new_feature_engineering + ordinal_mapping (aus src.features),
# ebenfalls nur auf Feature-Spalten (Id + SalePrice bleiben unverändert).
#
# Zusätzlich exportieren wir train/test als lokale Parquet-Dateien + manifest.json
# nach /Volumes/workspace/house_prices/feature_store (für lokalen Download per CLI).
#
# Output (Delta)
# ------------------------------------
#   - workspace.house_prices.gold_train_features
#   - workspace.house_prices.gold_test_features
#
# Output (Files)
# ------------------------------------
#   - /Volumes/workspace/house_prices/feature_store/train_gold.parquet
#   - /Volumes/workspace/house_prices/feature_store/test_gold.parquet
#   - /Volumes/workspace/house_prices/feature_store/manifest.json
# ------------------------------------

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from src.features import new_feature_engineering, ordinal_mapping

CATALOG = "workspace"
SCHEMA = "house_prices"

SILVER_TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.silver_train_clean"
SILVER_TEST_TABLE  = f"{CATALOG}.{SCHEMA}.silver_test_clean"

GOLD_TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.gold_train_features"
GOLD_TEST_TABLE  = f"{CATALOG}.{SCHEMA}.gold_test_features"

FEATURE_STORE_DIR = Path("/Volumes/workspace/house_prices/feature_store")
TRAIN_PARQUET = FEATURE_STORE_DIR / "train_gold.parquet"
TEST_PARQUET  = FEATURE_STORE_DIR / "test_gold.parquet"
MANIFEST_JSON = FEATURE_STORE_DIR / "manifest.json"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

silver_train = spark.table(SILVER_TRAIN_TABLE).toPandas()
silver_test  = spark.table(SILVER_TEST_TABLE).toPandas()

# Train
if "Id" not in silver_train.columns or "SalePrice" not in silver_train.columns:
    raise ValueError("Silver-Train muss 'Id' und 'SalePrice' enthalten.")

train_ids = silver_train[["Id"]]
train_y   = silver_train[["SalePrice"]]
X_train   = silver_train.drop(columns=["Id", "SalePrice"])

X_train = new_feature_engineering(X_train)
X_train = ordinal_mapping(X_train)

gold_train_pd = pd.concat([train_ids, train_y, X_train], axis=1)

# Test
if "Id" not in silver_test.columns:
    raise ValueError("Silver-Test muss 'Id' enthalten.")

test_ids = silver_test[["Id"]]
X_test   = silver_test.drop(columns=["Id"])

X_test = new_feature_engineering(X_test)
X_test = ordinal_mapping(X_test)

gold_test_pd = pd.concat([test_ids, X_test], axis=1)

gold_train_sdf = spark.createDataFrame(gold_train_pd)
gold_test_sdf  = spark.createDataFrame(gold_test_pd)

# Reset + overwriteSchema (Schema-Altlasten killen)
spark.sql(f"DROP TABLE IF EXISTS {GOLD_TRAIN_TABLE}")
spark.sql(f"DROP TABLE IF EXISTS {GOLD_TEST_TABLE}")

(
    gold_train_sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TRAIN_TABLE)
)

(
    gold_test_sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TEST_TABLE)
)

# Single-file Parquet exports (pandas)
gold_train_pd.to_parquet(TRAIN_PARQUET, index=False)
gold_test_pd.to_parquet(TEST_PARQUET, index=False)

manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "gold_train_table": GOLD_TRAIN_TABLE,
    "gold_test_table": GOLD_TEST_TABLE,
    "train_parquet": str(TRAIN_PARQUET),
    "test_parquet": str(TEST_PARQUET),
    "train_rows": int(gold_train_pd.shape[0]),
    "test_rows": int(gold_test_pd.shape[0]),
}

MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Gold tables created:")
print(f"- {GOLD_TRAIN_TABLE} | rows={gold_train_sdf.count()}")
print(f"- {GOLD_TEST_TABLE}  | rows={gold_test_sdf.count()}")

print("Gold parquet exported:")
print(f"- {TRAIN_PARQUET}")
print(f"- {TEST_PARQUET}")
print(f"- {MANIFEST_JSON}")