# Databricks notebook source
# 01_bronze_ingest

from __future__ import annotations

from pyspark.sql import SparkSession


CATALOG = "workspace"
SCHEMA = "house_prices"

TRAIN_CSV = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files/train.csv"
TEST_CSV = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files/test.csv"


spark = SparkSession.builder.getOrCreate()

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")


def read_csv(path: str):
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )


train_df = read_csv(TRAIN_CSV)
test_df = read_csv(TEST_CSV)

train_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_train_raw")
test_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_test_raw")

print("Bronze tables created:")
print(f"- {CATALOG}.{SCHEMA}.bronze_train_raw | rows={train_df.count()}")
print(f"- {CATALOG}.{SCHEMA}.bronze_test_raw  | rows={test_df.count()}")