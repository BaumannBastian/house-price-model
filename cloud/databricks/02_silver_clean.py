# Databricks notebook source
# 02_silver_clean

from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import SparkSession


CATALOG = "workspace"
SCHEMA = "house_prices"

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")


root = Path.cwd()
while root != root.parent and not (root / "src").exists():
    root = root.parent
sys.path.insert(0, str(root))

from src.features import missing_value_treatment


bronze_train = spark.table(f"{CATALOG}.{SCHEMA}.bronze_train_raw").toPandas()
bronze_test = spark.table(f"{CATALOG}.{SCHEMA}.bronze_test_raw").toPandas()

silver_train = missing_value_treatment(bronze_train)
silver_test = missing_value_treatment(bronze_test)

spark.createDataFrame(silver_train).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.silver_train_clean"
)
spark.createDataFrame(silver_test).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.silver_test_clean"
)

print("Silver tables created:")
print(f"- {CATALOG}.{SCHEMA}.silver_train_clean | rows={len(silver_train)}")
print(f"- {CATALOG}.{SCHEMA}.silver_test_clean  | rows={len(silver_test)}")