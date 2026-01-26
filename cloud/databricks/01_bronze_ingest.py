# ------------------------------------
# cloud/databricks/01_bronze_ingest.py
#
# Bronze-Ingest: train.csv / test.csv "raw" einlesen und als Delta speichern.
# Missing Values als NULLs, damit später keine str+int Fehler entstehen.
#
# Output (Delta)
# ------------------------------------
#   - workspace.house_prices.bronze_train_raw
#   - workspace.house_prices.bronze_test_raw
# ------------------------------------

import pandas as pd

CATALOG = "workspace"
SCHEMA = "house_prices"

RAW_TRAIN_CSV = "/Volumes/workspace/house_prices/raw_files/train.csv"
RAW_TEST_CSV  = "/Volumes/workspace/house_prices/raw_files/test.csv"

BRONZE_TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.bronze_train_raw"
BRONZE_TEST_TABLE  = f"{CATALOG}.{SCHEMA}.bronze_test_raw"

# Kaggle: "NA" taucht gelegentlich als Missing-Token auf (zusätzlich zu leeren Feldern)
NA_VALUES = ["NA", "N/A", "null", "None", ""]

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# pandas -> stabilere Typ-Inferenz für dieses kleine Dataset (und NA sauber als NaN)
train_pd = pd.read_csv(RAW_TRAIN_CSV, na_values=NA_VALUES, keep_default_na=True)
test_pd  = pd.read_csv(RAW_TEST_CSV,  na_values=NA_VALUES, keep_default_na=True)

train_sdf = spark.createDataFrame(train_pd)
test_sdf  = spark.createDataFrame(test_pd)

# Hard reset, damit Schema-Altlasten nicht reinfunken
spark.sql(f"DROP TABLE IF EXISTS {BRONZE_TRAIN_TABLE}")
spark.sql(f"DROP TABLE IF EXISTS {BRONZE_TEST_TABLE}")

(
    train_sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TRAIN_TABLE)
)

(
    test_sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TEST_TABLE)
)

print("Bronze tables created:")
print(f"- {BRONZE_TRAIN_TABLE} | rows={train_sdf.count()}")
print(f"- {BRONZE_TEST_TABLE}  | rows={test_sdf.count()}")