import sqlite3
import os
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def seed():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists
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

    demo_users = [
        ('user_001', 'farmer.pavir@gmail.com', generate_password_hash('Farmer@123'), 'Pavithran K', '+91 9876543210', 'Tamil Nadu', 'Erode'),
        ('user_002', 'ramesh.seri@gmail.com', generate_password_hash('RameshPass'), 'Ramesh Kumar', '+91 9123456789', 'Karnataka', 'Ramanagara'),
        ('user_003', 'anita.silk@gmail.com', generate_password_hash('Anita@2026'), 'Anita Devi', '+91 9988776655', 'Andhra Pradesh', 'Anantapur')
    ]

    for uid, email, p_hash, name, phone, state, district in demo_users:
        cursor.execute('''
            INSERT INTO users (uid, email, password_hash, full_name, phone, state, district)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                full_name = excluded.full_name,
                phone = excluded.phone,
                state = excluded.state,
                district = excluded.district
        ''', (uid, email, p_hash, name, phone, state, district))

    conn.commit()
    conn.close()
    print("[OK] Demo user records successfully inserted into users.db!")

if __name__ == '__main__':
    seed()
