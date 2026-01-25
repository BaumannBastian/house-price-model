# Databricks notebook source
# MAGIC %run ./00_bootstrap

# COMMAND ----------
exec(open(f"{REPO_ROOT}/cloud/databricks/01_bronze_ingest.py", "r", encoding="utf-8").read())