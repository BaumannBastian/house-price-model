# ------------------------------
# terraform/variables.tf
#
# In dieser Terraform-Datei werden Eingabevariablen für das
# House-Price-Projekt definiert (Projektname, Region, Umgebung,
# PostgreSQL-Parameter und Client-IP für die Firewall-Regel).
# ------------------------------

variable "project_name" {
  type        = string
  description = "Kurzname des Projekts, wird zur Benennung von Azure-Ressourcen benutzt."
  default     = "house-price"
}

variable "location" {
  type        = string
  description = "Azure-Region für Resource Group und PostgreSQL Flexible Server."
  default     = "northeurope"
}

variable "environment" {
  type        = string
  description = "Umgebung (z. B. dev, prod)."
  default     = "dev"
}

variable "db_name" {
  type        = string
  description = "Name der PostgreSQL-Datenbank."
  default     = "house_prices"
}

variable "db_admin_username" {
  type        = string
  description = "Administrator-Benutzername für den PostgreSQL Flexible Server."
  default     = "hpadmin"
}

variable "postgres_version" {
  type        = string
  description = "PostgreSQL-Version für den Flexible Server (z. B. 16)."
  default     = "16"
}

variable "client_ip" {
  type        = string
  description = "Öffentliche IPv4-Adresse des lokalen Clients, der auf die DB zugreifen darf (z. B. 84.138.160.151)."
}