import os
import sqlite3
import pandas as pd
import json

def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path):
    """
    Initializes the SQLite tables for incidents and dispatch logs.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Create incidents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            classification TEXT,
            type TEXT,
            severity TEXT,
            confidence REAL,
            camera_id TEXT,
            location TEXT,
            lat REAL,
            lon REAL,
            dispatch_status TEXT,
            explanation TEXT,
            vehicles_involved TEXT,
            near_miss_score INTEGER,
            weather TEXT,
            traffic_density TEXT
        )
    """)
    
    # Create dispatch_logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dispatch_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera_id TEXT,
            type TEXT,
            recipient TEXT,
            status TEXT,
            details TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def migrate_json_data(db_path, history_file, dispatch_log_file):
    """
    Migrates records from JSON files to SQLite database if database is empty.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Check if we already have records
    cursor.execute("SELECT COUNT(*) FROM incidents")
    has_incidents = cursor.fetchone()[0] > 0
    
    if not has_incidents and os.path.exists(history_file):
        try:
            df = pd.read_json(history_file)
            if not df.empty:
                # Replace date column if present and not needed, fill nulls
                if "date" in df.columns:
                    df = df.drop(columns=["date"])
                
                # Fill missing columns
                for col in ["explanation", "vehicles_involved", "near_miss_score", "weather", "traffic_density"]:
                    if col not in df.columns:
                        df[col] = None
                
                # Write to sqlite
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT OR IGNORE INTO incidents (
                            id, timestamp, classification, type, severity, confidence, 
                            camera_id, location, lat, lon, dispatch_status, 
                            explanation, vehicles_involved, near_miss_score, weather, traffic_density
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get("id"),
                        str(row.get("timestamp")),
                        row.get("classification"),
                        row.get("type"),
                        row.get("severity"),
                        float(row.get("confidence")) if pd.notnull(row.get("confidence")) else 0.0,
                        row.get("camera_id"),
                        row.get("location"),
                        float(row.get("lat")) if pd.notnull(row.get("lat")) else None,
                        float(row.get("lon")) if pd.notnull(row.get("lon")) else None,
                        row.get("dispatch_status"),
                        row.get("explanation"),
                        row.get("vehicles_involved"),
                        int(row.get("near_miss_score")) if pd.notnull(row.get("near_miss_score")) else 0,
                        row.get("weather"),
                        row.get("traffic_density")
                    ))
                print(f"Migrated {len(df)} incidents from JSON to SQLite.")
        except Exception as e:
            print("Error migrating history JSON:", e)
            
    # Migrate dispatch logs
    cursor.execute("SELECT COUNT(*) FROM dispatch_logs")
    has_dispatch = cursor.fetchone()[0] > 0
    
    if not has_dispatch and os.path.exists(dispatch_log_file):
        try:
            with open(dispatch_log_file, "r") as f:
                logs = json.load(f)
            if logs:
                for log in logs:
                    cursor.execute("""
                        INSERT INTO dispatch_logs (timestamp, camera_id, type, recipient, status, details)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        log.get("timestamp"),
                        log.get("camera_id"),
                        log.get("type"),
                        log.get("recipient"),
                        log.get("status"),
                        log.get("details")
                    ))
                print(f"Migrated {len(logs)} dispatch logs from JSON to SQLite.")
        except Exception as e:
            print("Error migrating dispatch logs JSON:", e)
            
    conn.commit()
    conn.close()

def save_incident(db_path, incident_data):
    """
    Saves or updates an incident in the SQLite database.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO incidents (
            id, timestamp, classification, type, severity, confidence, 
            camera_id, location, lat, lon, dispatch_status, 
            explanation, vehicles_involved, near_miss_score, weather, traffic_density
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        incident_data.get("id"),
        incident_data.get("timestamp"),
        incident_data.get("classification"),
        incident_data.get("type"),
        incident_data.get("severity"),
        incident_data.get("confidence"),
        incident_data.get("camera_id"),
        incident_data.get("location"),
        incident_data.get("lat"),
        incident_data.get("lon"),
        incident_data.get("dispatch_status"),
        incident_data.get("explanation"),
        incident_data.get("vehicles_involved"),
        incident_data.get("near_miss_score"),
        incident_data.get("weather"),
        incident_data.get("traffic_density")
    ))
    conn.commit()
    conn.close()

def update_dispatch_status(db_path, incident_id, status_str):
    """
    Updates the dispatch status of a specific incident.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE incidents SET dispatch_status = ? WHERE id = ?", (status_str, incident_id))
    conn.commit()
    conn.close()

def save_dispatch_log(db_path, log_data):
    """
    Saves a transmission log to SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO dispatch_logs (timestamp, camera_id, type, recipient, status, details)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        log_data.get("timestamp"),
        log_data.get("camera_id"),
        log_data.get("type"),
        log_data.get("recipient"),
        log_data.get("status"),
        log_data.get("details")
    ))
    conn.commit()
    conn.close()

def load_incidents(db_path):
    """
    Loads all incidents from the SQLite database as a Pandas DataFrame.
    """
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def load_dispatch_logs(db_path):
    """
    Loads all dispatch logs from the SQLite database as a list of dicts.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dispatch_logs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def clear_database(db_path):
    """
    Deletes all records from both tables.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents")
    cursor.execute("DELETE FROM dispatch_logs")
    conn.commit()
    conn.close()
