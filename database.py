import sqlite3
import hashlib
import json
from datetime import datetime

DB_NAME = "content_store.db"
# database.py

def setup_database():
    """Create the database and necessary table if not exists."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS content (
        id TEXT PRIMARY KEY,
        url TEXT,
        raw_content TEXT,
        metadata TEXT,
        tags TEXT,
        summary TEXT,
        normalized_content TEXT,
        vectorized_content TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

# database.py

def add_normalized_content_column():
    """Add normalized_content column to the content table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(content)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'normalized_content' not in columns:
        cursor.execute("ALTER TABLE content ADD COLUMN normalized_content TEXT")
    conn.commit()
    conn.close()

def store_in_db(url, raw_content, metadata, tags, summary, normalized_content, vectorized_content):
    """Store processed content in SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Generate a unique ID based on content
    content_id = hashlib.sha256((url or raw_content).encode()).hexdigest()

    # Serialize vectorized content (embedding) as JSON
    vectorized_content_json = json.dumps(vectorized_content or [])

    cursor.execute("""
    INSERT OR REPLACE INTO content (id, url, raw_content, metadata, tags, summary, normalized_content, vectorized_content, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (content_id, url, raw_content, metadata, tags, summary, normalized_content, vectorized_content_json, datetime.now().isoformat()))

    conn.commit()
    conn.close()

