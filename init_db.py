import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT UNIQUE,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            full_name TEXT,
            phone TEXT,
            state TEXT,
            district TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized successfully at: {DB_PATH}")

if __name__ == '__main__':
    create_database()
