import os
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL
                    )
                ''')
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS scans (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) NOT NULL,
                        label VARCHAR(255) NOT NULL,
                        confidence REAL NOT NULL,
                        top_k TEXT,
                        image_path TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (username) REFERENCES users(username)
                    )
                ''')
                # Ensure new columns exist for older tables
                cur.execute('ALTER TABLE scans ADD COLUMN IF NOT EXISTS matched_path TEXT')
                cur.execute('ALTER TABLE scans ADD COLUMN IF NOT EXISTS matched_label TEXT')
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (username) REFERENCES users(username)
                    )
                ''')
            conn.commit()
    except Exception as e:
        print(f"Error initializing DB: {e}")

def save_scan(username, label, confidence, top_k, image_path, matched_path=None, matched_label=None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO scans (username, label, confidence, top_k, image_path, matched_path, matched_label) VALUES (%s, %s, %s, %s, %s, %s, %s)',
                (username, label, confidence, json.dumps(top_k), image_path, matched_path, matched_label)
            )
        conn.commit()

def get_history(username):
    with get_db() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                'SELECT * FROM scans WHERE username = %s ORDER BY timestamp DESC LIMIT 20',
                (username,)
            )
            return cur.fetchall()

def create_user(username, hashed_password):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO users (username, password) VALUES (%s, %s)',
                (username, hashed_password)
            )
        conn.commit()

def get_user(username):
    with get_db() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                'SELECT * FROM users WHERE username = %s',
                (username,)
            )
            return cur.fetchone()

def delete_scan(scan_id, username):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'DELETE FROM scans WHERE id = %s AND username = %s',
                (scan_id, username)
            )
        conn.commit()

if __name__ == '__main__':
    init_db()
    print("PostgreSQL Database initialized.")
