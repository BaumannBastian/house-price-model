# ------------------------------------
# cloud/databricks/03_gold_features.py
#
# Erzeugt den GOLD Layer:
# - Feature Engineering auf Basis der SILVER Tabellen
# - Schreibt Gold-Delta-Tabellen (Unity Catalog)
# - Exportiert train/test als Parquet in den Volume "feature_store"
#
# Output (Parquet):
#   /Volumes/workspace/house_prices/feature_store/train_gold.parquet
#   /Volumes/workspace/house_prices/feature_store/test_gold.parquet
# ------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Projekt-Repo-Root finden, damit `src.*` importierbar ist
repo_root = Path.cwd()
while repo_root != repo_root.parent and not (repo_root / "src").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.features import apply_feature_engineering  # noqa: E402


CATALOG = "workspace"
SCHEMA = "house_prices"

SILVER_TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.silver_train_clean"
SILVER_TEST_TABLE = f"{CATALOG}.{SCHEMA}.silver_test_clean"

GOLD_TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.gold_train_features"
GOLD_TEST_TABLE = f"{CATALOG}.{SCHEMA}.gold_test_features"

# Unity Catalog Volume (tabular/feature output als File)
VOLUME_NAME = "feature_store"
VOLUME_DBFS_BASE = f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
VOLUME_DRIVER_BASE = f"/dbfs/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"


def main() -> None:
    # 1) SILVER lesen
    silver_train = spark.read.table(SILVER_TRAIN_TABLE).toPandas()
    silver_test = spark.read.table(SILVER_TEST_TABLE).toPandas()

    # 2) Feature Engineering (Pandas, wie lokal)
    gold_train = apply_feature_engineering(silver_train)
    gold_test = apply_feature_engineering(silver_test)

    # 3) GOLD Delta-Tabellen schreiben (Unity Catalog)
    spark.createDataFrame(gold_train).write.mode("overwrite").saveAsTable(GOLD_TRAIN_TABLE)
    spark.createDataFrame(gold_test).write.mode("overwrite").saveAsTable(GOLD_TEST_TABLE)

    # 4) Parquet in Volume schreiben (als echte Dateien, nicht als Spark-Ordner)
    out_dir = Path(VOLUME_DRIVER_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_gold.parquet"
    test_path = out_dir / "test_gold.parquet"

    gold_train.to_parquet(train_path.as_posix(), index=False)
    gold_test.to_parquet(test_path.as_posix(), index=False)

    print("OK GOLD fertig.")
    print(f"- Delta: {GOLD_TRAIN_TABLE}, {GOLD_TEST_TABLE}")
    print(f"- Parquet: {VOLUME_DBFS_BASE}/train_gold.parquet")
    print(f"- Parquet: {VOLUME_DBFS_BASE}/test_gold.parquet")


if __name__ == "__main__":
    main()