
import mysql.connector
import sqlite3
import time
import os
import sys
import logging
from datetime import datetime

# Get absolute path to BIO/ and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_PORT, SQLITE_DB_PATH

logger = logging.getLogger("Database")

class LocalDatabase:
    def __init__(self):
        self.sqlite_path = SQLITE_DB_PATH
        self._init_sqlite()

    def _init_sqlite(self):
        """Initialize local SQLite database for offline buffering"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS local_attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    device_id TEXT,
                    name TEXT,
                    synced INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
            conn.close()
            logger.info(f"Local SQLite initialized at {self.sqlite_path}")
        except Exception as e:
            logger.error(f"Failed to init SQLite: {e}")

    def log_attendance(self, device_id, name):
        """Log attendance to local SQLite (Always works)"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            timestamp = time.time()
            
            cursor.execute('''
                INSERT INTO local_attendance (timestamp, device_id, name, synced)
                VALUES (?, ?, ?, 0)
            ''', (timestamp, device_id, name))
            
            conn.commit()
            conn.close()
            logger.info(f"Logged offline: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to log locally: {e}")
            return False

    def sync_to_mysql(self):
        """Sync unsynced records from SQLite to MySQL"""
        # 1. Get unsynced records
        records = []
        try:
            s_conn = sqlite3.connect(self.sqlite_path)
            s_cursor = s_conn.cursor()
            s_cursor.execute("SELECT id, timestamp, device_id, name FROM local_attendance WHERE synced = 0 LIMIT 50")
            records = s_cursor.fetchall()
            s_conn.close()
        except Exception as e:
            logger.error(f"Read error: {e}")
            return

        if not records:
            return # Nothing to sync

        # 2. Connect to MySQL
        m_conn = None
        try:
            m_conn = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB,
                port=MYSQL_PORT,
                connection_timeout=2 # Fast fail if offline
            )
        except:
             return # Offline

        # 3. Upload
        try:
            m_cursor = m_conn.cursor()
            synced_ids = []
            
            for row in records:
                id, ts, dev, name_val = row
                # Format timestamp for MySQL DATETIME
                dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                
                m_cursor.execute('''
                    INSERT INTO attendance_log (timestamp, device_id, name, synced)
                    VALUES (%s, %s, %s, 1)
                ''', (dt, dev, name_val))
                
                synced_ids.append(id)
            
            m_conn.commit()
            m_conn.close()
            
            # 4. Mark as Synced in SQLite
            if synced_ids:
                s_conn = sqlite3.connect(self.sqlite_path)
                s_cursor = s_conn.cursor()
                # Bulk update
                s_cursor.executemany("UPDATE local_attendance SET synced = 1 WHERE id = ?", [(i,) for i in synced_ids])
                s_conn.commit()
                s_conn.close()
                
                logger.info(f"Synced {len(synced_ids)} records to MySQL")
                
        except Exception as e:
            logger.error(f"Sync error: {e}")
            if m_conn: m_conn.close()
