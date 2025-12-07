# ------------------------------
# src/db.py
#
# In dieser Python-Datei werden Hilfsfunktionen bereitgestellt, um
# Verbindungen zur PostgreSQL-Datenbank herzustellen sowie
# Vorhersagen und Modell-Metadaten zu speichern und auszulesen.
#
# Die Verbindungsdaten (Host, Port, Name, User, Passwort, SSL-Mode)
# werden über Umgebungsvariablen konfiguriert:
#
#   DB_HOST     – z.B. FQDN des Azure-Postgres-Servers
#   DB_PORT     – z.B. "5432"
#   DB_NAME     – z.B. "house_prices"
#   DB_USER     – z.B. "hpadmin" oder ein eigener App-User
#   DB_PASSWORD – Passwort für diesen User
#   DB_SSLMODE  – z.B. "require" (Azure) oder "disable" (lokale Tests)
#
# Für lokale Entwicklung existieren sinnvolle Default-Werte, sodass
# die Variablen nicht zwingend gesetzt sein müssen.
# ------------------------------

import os
from typing import Iterable

import numpy as np
import psycopg2
from psycopg2.extras import Json


# ---------------------------------------------------------------------------
# Konfiguration über Umgebungsvariablen
# ---------------------------------------------------------------------------

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "house_prices")
DB_USER = os.getenv("DB_USER", "house")
DB_PASSWORD = os.getenv("DB_PASSWORD", "house")
# Für Azure-Postgres ist "require" üblich; für lokale Docker-DB kann
# man DB_SSLMODE="disable" setzen.
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")


def get_connection():
    """
    Stellt eine neue Verbindung zur PostgreSQL-Datenbank her.

    Die Verbindungsparameter werden über Umgebungsvariablen konfiguriert
    und in Modulkonstanten gespiegelt:

    - ``DB_HOST`` (str): Hostname oder FQDN des Servers (z.B. Azure-Postgres).
    - ``DB_PORT`` (int): Portnummer, standardmäßig 5432.
    - ``DB_NAME`` (str): Name der Datenbank (z.B. ``"house_prices"``).
    - ``DB_USER`` (str): Datenbank-Benutzername.
    - ``DB_PASSWORD`` (str): Passwort des Benutzers.
    - ``DB_SSLMODE`` (str): SSL-Modus, z.B. ``"require"`` (Azure)
      oder ``"disable"`` (lokal ohne SSL).

    Wird keine Umgebungsvariable gesetzt, greifen Default-Werte
    (lokaler PostgreSQL mit User/Passwort ``house`` und DB
    ``house_prices``).

    Returns
    -------
    psycopg2.extensions.connection
        Geöffnete Datenbankverbindung. Der Aufrufer ist dafür
        verantwortlich, die Verbindung nach der Benutzung mit
        ``conn.close()`` wieder zu schließen.
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
    )
    return conn


def init_db():
    """
    Initialisiert die Tabelle ``predictions`` in der Datenbank.

    Die Funktion legt die Tabelle ``predictions`` an, falls sie noch
    nicht existiert, und stellt sicher, dass eine Spalte ``model_id``
    vorhanden ist. Die Änderungen werden direkt in der Datenbank
    committet.

    Returns
    -------
    None
        Diese Funktion gibt keinen Wert zurück. Bei Erfolg existiert
        anschließend eine einsatzbereite Tabelle ``predictions``.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        kaggle_id INTEGER NOT NULL,
        predicted_price DOUBLE PRECISION NOT NULL,
        model_id INTEGER,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    # Für den Fall, dass die Tabelle schon ohne model_id existiert:
    alter_add_model_id_sql = """
    ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS model_id INTEGER;
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                cur.execute(alter_add_model_id_sql)
        print("Tabelle 'predictions' ist bereit (inkl. model_id).")
    finally:
        conn.close()


def insert_prediction(kaggle_id: int, predicted_price: float) -> None:
    """
    Fügt eine einzelne Vorhersage in die Tabelle ``predictions`` ein.

    Diese Funktion ist praktisch für einfache Tests oder Einzelfälle,
    in denen nur eine einzelne Zeile in der Datenbank gespeichert
    werden soll.

    Parameter
    ----------
    kaggle_id : int
        ID aus dem Kaggle-Testdatensatz (Spalte ``Id`` in ``test.csv``).
    predicted_price : float
        Vorhergesagter Verkaufspreis (im Originalraum, nicht geloggt).

    Returns
    -------
    None
        Es wird kein Wert zurückgegeben. Die neue Zeile wird in
        der Datenbank persistiert.
    """
    insert_sql = """
    INSERT INTO predictions (kaggle_id, predicted_price)
    VALUES (%s, %s);
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (int(kaggle_id), float(predicted_price)))
    finally:
        conn.close()


def insert_predictions(
    kaggle_ids: Iterable[int],
    predicted_prices: Iterable[float],
    model_id: int | None = None,
) -> int:
    """
    Fügt mehrere Vorhersagen in einem Schritt in die Tabelle ``predictions`` ein.

    Die übergebenen ``kaggle_ids`` und ``predicted_prices`` werden
    paarweise verknüpft und in einem gemeinsamen Datenbank-Commit
    gespeichert. Optional kann pro Zeile eine ``model_id`` mitgegeben
    werden, die auf einen Eintrag in der Tabelle ``models`` verweist.

    Parameter
    ----------
    kaggle_ids : Iterable[int]
        Iterable von Kaggle-IDs (z.B. ``test_df["Id"]``).
    predicted_prices : Iterable[float]
        Iterable von vorhergesagten Hauspreisen in exakt derselben
        Reihenfolge wie ``kaggle_ids``.
    model_id : int | None, optional
        Fremdschlüssel auf ``models.id``, der das verwendete Modell
        referenziert. Standard ist ``None`` (keine Modellreferenz).

    Returns
    -------
    int
        Anzahl der tatsächlich eingefügten Zeilen. Gibt ``0`` zurück,
        wenn keine Paare übergeben wurden.
    """
    insert_sql = """
    INSERT INTO predictions (kaggle_id, predicted_price, model_id)
    VALUES (%s, %s, %s);
    """

    pairs = [
        (int(k_id), float(pred), model_id)
        for k_id, pred in zip(kaggle_ids, predicted_prices)
    ]

    if not pairs:
        return 0

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, pairs)
        return len(pairs)
    finally:
        conn.close()


def fetch_last_predictions(limit: int = 5):
    """
    Holt die letzten ``limit`` Vorhersagen aus der Tabelle ``predictions``.

    Die Vorhersagen werden absteigend nach ``created_at`` sortiert,
    sodass der neueste Eintrag zuerst erscheint.

    Parameter
    ----------
    limit : int, optional
        Maximale Anzahl der zurückzugebenden Vorhersagen. Standard ist ``5``.

    Returns
    -------
    list[tuple[int, float, datetime.datetime]]
        Liste von Tupeln der Form ``(kaggle_id, predicted_price, created_at)``.
        Die genaue Typinformation von ``created_at`` hängt vom verwendeten
        psycopg2-Typadapter ab, ist aber in der Regel ein
        ``datetime.datetime``-Objekt.
    """
    select_sql = """
    SELECT kaggle_id, predicted_price, created_at
    FROM predictions
    ORDER BY created_at DESC
    LIMIT %s;
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(select_sql, (limit,))
                return cur.fetchall()
    finally:
        conn.close()


# Modelle-Tabelle für Metadaten
def init_models_table():
    """
    Initialisiert die Tabelle ``models`` für Modell-Metadaten.

    Die Tabelle ``models`` speichert u.a. den Modellnamen, eine Versions-
    information, den Pfad zur Modell-Datei, verschiedene Testmetriken
    sowie Hyperparameter als JSONB. Zusätzlich wird ein Flag
    ``is_champion`` gepflegt, um das aktuell bevorzugte Modell zu markieren.

    Returns
    -------
    None
        Diese Funktion gibt keinen Wert zurück. Nach erfolgreicher
        Ausführung existiert die Tabelle ``models`` in der Datenbank
        (falls sie nicht bereits vorhanden war).
    """
    sql = """
        CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            file_path TEXT NOT NULL,
            r2_test DOUBLE PRECISION,
            rmse_test DOUBLE PRECISION,
            mare_test DOUBLE PRECISION,
            cv_rmse_mean DOUBLE PRECISION,
            cv_rmse_std DOUBLE PRECISION,
            hyperparams JSONB,
            is_champion BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
    finally:
        conn.close()


def insert_model(
    name: str,
    version: str,
    file_path: str,
    r2_test,
    rmse_test,
    mare_test,
    cv_rmse_mean,
    cv_rmse_std,
    is_champion: bool,
    hyperparams: dict | None = None,
) -> int:
    """
    Fügt ein Modell in die Tabelle ``models`` ein und gibt die neue ID zurück.

    Die wichtigsten Testmetriken (R², RMSE, MARE, CV-RMSE-Mittelwert und
    -Standardabweichung) werden auf normale Python-Floats gecastet, um
    Probleme mit NumPy-Datentypen beim Speichern in PostgreSQL zu vermeiden.
    Hyperparameter werden als JSONB abgelegt, nachdem sie rekursiv auf
    einfache Python-Typen (float, bool, list, dict, …) abgebildet wurden.

    Parameter
    ----------
    name : str
        Logischer Name des Modells, z.B. ``"HistGBR_log"``.
    version : str
        Versionsstring des Modells, z.B. eine SemVer-Version oder ein
        Git-Commit-Hash.
    file_path : str
        Pfad zur gespeicherten Modell-Datei (z.B. Pickle oder Joblib).
    r2_test
        R² auf dem Testdatensatz (wird intern auf ``float`` gecastet).
    rmse_test
        RMSE auf dem Testdatensatz (wird intern auf ``float`` gecastet).
    mare_test
        MARE auf dem Testdatensatz (wird intern auf ``float`` gecastet).
    cv_rmse_mean
        Mittlerer RMSE über Cross-Validation-Folds (wird auf ``float``
        gecastet).
    cv_rmse_std
        Standardabweichung des RMSE über Cross-Validation-Folds (wird auf
        ``float`` gecastet).
    is_champion : bool
        Flag, ob dieses Modell aktuell der Champion ist (``True``) oder nicht.
    hyperparams : dict | None, optional
        Dictionary mit Hyperparametern des Modells. Die Werte dürfen auch
        NumPy-Typen enthalten; sie werden rekursiv auf einfache Python-Typen
        abgebildet und als JSONB gespeichert. Standard ist ein leeres Dict.

    Returns
    -------
    int
        Primärschlüssel ``id`` des neu eingefügten Eintrags in ``models``.
    """
    sql = """
    INSERT INTO models (
        name, version, file_path,
        r2_test, rmse_test, mare_test,
        cv_rmse_mean, cv_rmse_std,
        hyperparams,
        is_champion
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """

    # Helper, um numpy-Typen loszuwerden
    def to_plain_number(x):
        if x is None:
            return None
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return float(x)

    # numerische Metriken auf float casten
    r2_test = to_plain_number(r2_test) if r2_test is not None else None
    rmse_test = to_plain_number(rmse_test) if rmse_test is not None else None
    mare_test = to_plain_number(mare_test) if mare_test is not None else None
    cv_rmse_mean = to_plain_number(cv_rmse_mean) if cv_rmse_mean is not None else None
    cv_rmse_std = to_plain_number(cv_rmse_std) if cv_rmse_std is not None else None

    # hyperparams auf normale Python-Typen bringen
    hyperparams = hyperparams or {}

    def clean_value(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (list, tuple)):
            return [clean_value(x) for x in v]
        if isinstance(v, dict):
            return {k: clean_value(x) for k, x in v.items()}
        return v

    hyper_clean = {k: clean_value(v) for k, v in hyperparams.items()}

    # Json(...) sorgt dafür, dass Postgres das sauber als JSONB bekommt
    hyper_param = Json(hyper_clean)

    params = (
        str(name),
        str(version),
        str(file_path),
        r2_test,
        rmse_test,
        mare_test,
        cv_rmse_mean,
        cv_rmse_std,
        hyper_param,
        bool(is_champion),
    )

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                new_id = cur.fetchone()[0]
        return new_id
    finally:
        conn.close()


def get_current_champion_id() -> int | None:
    """
    Liefert die ID des aktuell als Champion markierten Modells.

    Es wird nach Einträgen in ``models`` mit ``is_champion = TRUE``
    gesucht. Falls mehrere Champions existieren, wird der zuletzt
    erstellte Eintrag (nach ``created_at``) zurückgegeben.

    Returns
    -------
    int | None
        Die ``id`` des zuletzt eingetragenen Champion-Modells oder
        ``None``, falls kein Champion in der Tabelle ``models`` vorhanden ist.
    """
    sql = """
    SELECT id
    FROM models
    WHERE is_champion = TRUE
    ORDER BY created_at DESC
    LIMIT 1;
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
                if row is None:
                    return None
                return row[0]
    finally:
        conn.close()