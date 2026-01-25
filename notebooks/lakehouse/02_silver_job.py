# Databricks notebook source
# MAGIC %run ./00_bootstrap

# COMMAND ----------
exec(open(f"{REPO_ROOT}/cloud/databricks/02_silver_clean.py", "r", encoding="utf-8").read())