# ------------------------------------
# cloud/databricks/02_silver_clean.py
#
# Silver-Clean: missing_value_treatment (aus src.features),
# aber nur auf den Feature-Spalten (Id + SalePrice bleiben unverändert).
#
# Output (Delta)
# ------------------------------------
#   - workspace.house_prices.silver_train_clean
#   - workspace.house_prices.silver_test_clean
# ------------------------------------

import pandas as pd
from src.features import missing_value_treatment

CATALOG = "workspace"
SCHEMA = "house_prices"

BRONZE_TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.bronze_train_raw"
BRONZE_TEST_TABLE  = f"{CATALOG}.{SCHEMA}.bronze_test_raw"

SILVER_TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.silver_train_clean"
SILVER_TEST_TABLE  = f"{CATALOG}.{SCHEMA}.silver_test_clean"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Load bronze
bronze_train = spark.table(BRONZE_TRAIN_TABLE).toPandas()
bronze_test  = spark.table(BRONZE_TEST_TABLE).toPandas()

# Train: Id + SalePrice separat halten
if "Id" not in bronze_train.columns or "SalePrice" not in bronze_train.columns:
    raise ValueError("Bronze-Train muss 'Id' und 'SalePrice' enthalten.")

train_ids = bronze_train[["Id"]]
train_y   = bronze_train[["SalePrice"]]
X_train   = bronze_train.drop(columns=["Id", "SalePrice"])

# Test: Id separat halten
if "Id" not in bronze_test.columns:
    raise ValueError("Bronze-Test muss 'Id' enthalten.")

test_ids = bronze_test[["Id"]]
X_test   = bronze_test.drop(columns=["Id"])

# missing_value_treatment nur auf Features
X_train_clean = missing_value_treatment(X_train)
X_test_clean  = missing_value_treatment(X_test)

silver_train_pd = pd.concat([train_ids, train_y, X_train_clean], axis=1)
silver_test_pd  = pd.concat([test_ids, X_test_clean], axis=1)

silver_train_sdf = spark.createDataFrame(silver_train_pd)
silver_test_sdf  = spark.createDataFrame(silver_test_pd)

# Schema-Konflikte killen (LotFrontage etc.)
spark.sql(f"DROP TABLE IF EXISTS {SILVER_TRAIN_TABLE}")
spark.sql(f"DROP TABLE IF EXISTS {SILVER_TEST_TABLE}")

(
    silver_train_sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TRAIN_TABLE)
)

(
    silver_test_sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TEST_TABLE)
)

print("Silver tables created:")
print(f"- {SILVER_TRAIN_TABLE} | rows={silver_train_sdf.count()}")
print(f"- {SILVER_TEST_TABLE}  | rows={silver_test_sdf.count()}")