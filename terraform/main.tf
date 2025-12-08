# ------------------------------
# terraform/main.tf
#
# In dieser Terraform-Datei werden Provider, eine Resource Group
# und ein Azure Database for PostgreSQL – Flexible Server für das
# House-Price-Projekt definiert (inkl. Firewall-Regel und Datenbank).
# ------------------------------

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}

  # Explizit die Subscription-ID setzen (erforderlich ab Provider v4)
  subscription_id = "41633924-21c7-41f8-aa27-0893c49d41ab"
}

# Resource Group
resource "azurerm_resource_group" "rg" {
  name     = "${var.project_name}-rg"
  location = var.location
}

# Admin-Passwort sicher generieren
resource "random_password" "db_admin_password" {
  length           = 24
  special          = true
  override_special = "!@#-_"
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "db" {
  name                = "${var.project_name}-psql"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location

  administrator_login    = var.db_admin_username
  administrator_password = random_password.db_admin_password.result

  sku_name = "B_Standard_B1ms" # Burstable, 1 vCore (Free-Tier-tauglich, Details siehe Pricing)
  version  = var.postgres_version

  storage_mb                    = 32768
  backup_retention_days         = 7
  storage_tier                  = "P4"
  auto_grow_enabled             = false
  geo_redundant_backup_enabled  = false
  public_network_access_enabled = true

  zone = "1"

  tags = {
    project     = var.project_name
    environment = var.environment
  }
}

# Anwendungs-Datenbank
resource "azurerm_postgresql_flexible_server_database" "app_db" {
  name      = var.db_name
  server_id = azurerm_postgresql_flexible_server.db.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# Firewall-Regel: nur deine aktuelle öffentliche IP
resource "azurerm_postgresql_flexible_server_firewall_rule" "local_dev" {
  name      = "allow-local-dev-ip"
  server_id = azurerm_postgresql_flexible_server.db.id

  start_ip_address = var.client_ip
  end_ip_address   = var.client_ip
}