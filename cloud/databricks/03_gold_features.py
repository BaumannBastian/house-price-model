# Databricks notebook source
# 03_gold_features

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession

CATALOG = "workspace"
SCHEMA = "house_prices"

# DBFS Volume (für lokalen Download via databricks CLI)
VOLUME_DIR = Path("/dbfs/Volumes/workspace/house_prices/feature_store")
TRAIN_PARQUET_NAME = "train_gold.parquet"
TEST_PARQUET_NAME = "test_gold.parquet"
MANIFEST_NAME = "manifest.json"

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Repo-Root finden und src/ importierbar machen
root = Path.cwd()
while root != root.parent and not (root / "src").exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.features import new_feature_engineering, ordinal_mapping  # noqa: E402


# -----------------------------
# Load Silver (Delta -> Pandas)
# -----------------------------
silver_train = spark.table(f"{CATALOG}.{SCHEMA}.silver_train_clean").toPandas()
silver_test = spark.table(f"{CATALOG}.{SCHEMA}.silver_test_clean").toPandas()

# -----------------------------
# Feature Engineering (Pandas)
# -----------------------------
gold_train = ordinal_mapping(new_feature_engineering(silver_train))
gold_test = ordinal_mapping(new_feature_engineering(silver_test))

# -----------------------------
# Save Gold as Delta tables
# -----------------------------
(
    spark.createDataFrame(gold_train)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_train_features")
)
(
    spark.createDataFrame(gold_test)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_test_features")
)

# -----------------------------
# Save Gold as Parquet for local training
# -----------------------------
VOLUME_DIR.mkdir(parents=True, exist_ok=True)

train_path = VOLUME_DIR / TRAIN_PARQUET_NAME
test_path = VOLUME_DIR / TEST_PARQUET_NAME

gold_train.to_parquet(train_path, index=False)
gold_test.to_parquet(test_path, index=False)

# -----------------------------
# Write manifest (version/stamp)
# -----------------------------
manifest = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "catalog": CATALOG,
    "schema": SCHEMA,
    "volume_dir": "/Volumes/workspace/house_prices/feature_store",
    "files": {
        "train": TRAIN_PARQUET_NAME,
        "test": TEST_PARQUET_NAME,
    },
    "rows": {
        "train": int(len(gold_train)),
        "test": int(len(gold_test)),
    },
    "columns": {
        "train": list(gold_train.columns),
        "test": list(gold_test.columns),
    },
}

with open(VOLUME_DIR / MANIFEST_NAME, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("Gold tables created:")
print(f"- {CATALOG}.{SCHEMA}.gold_train_features | rows={len(gold_train)}")
print(f"- {CATALOG}.{SCHEMA}.gold_test_features  | rows={len(gold_test)}")
print("Gold parquet exported to volume:")
print(f"- {train_path}")
print(f"- {test_path}")
print(f"- {VOLUME_DIR / MANIFEST_NAME}")