# ------------------------------
# terraform/outputs.tf
#
# In dieser Terraform-Datei werden wichtige Ausgaben für den
# PostgreSQL Flexible Server bereitgestellt (FQDN, Admin-User,
# Datenbankname und ein Beispiel-Connection-String für psycopg2).
# ------------------------------

output "postgres_fqdn" {
  description = "FQDN des PostgreSQL Flexible Servers."
  value       = azurerm_postgresql_flexible_server.db.fqdn
}

output "postgres_admin_username" {
  description = "Administrator-Benutzername für PostgreSQL."
  value       = var.db_admin_username
}

output "postgres_admin_password" {
  description = "Generiertes Admin-Passwort für PostgreSQL (sensitiv!)."
  value       = random_password.db_admin_password.result
  sensitive   = true
}

output "postgres_database_name" {
  description = "Name der PostgreSQL-Datenbank."
  value       = var.db_name
}

output "psycopg2_connection_example" {
  description = "Beispiel-Connection-String für psycopg2."
  value       = "postgresql://${var.db_admin_username}:${random_password.db_admin_password.result}@${azurerm_postgresql_flexible_server.db.fqdn}:5432/${var.db_name}"
  sensitive   = true
}