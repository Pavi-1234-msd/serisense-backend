import os
import sys
from werkzeug.security import generate_password_hash

# Try mysql.connector or pymysql
try:
    import mysql.connector
    driver = 'mysql.connector'
except ImportError:
    try:
        import pymysql
        pymysql.install_as_MySQLdb()
        import MySQLdb as mysql.connector
        driver = 'pymysql'
    except ImportError:
        print("[ERROR] Neither mysql-connector-python nor pymysql is installed.")
        print("Please run: pip install mysql-connector-python")
        sys.exit(1)

MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
MYSQL_DB = os.environ.get('MYSQL_DB', 'serisense_db')
MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))

def init_mysql():
    print(f"Connecting to MySQL server at {MYSQL_HOST}:{MYSQL_PORT} (User: {MYSQL_USER})...")
    try:
        # Step 1: Connect to MySQL server & create database
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            port=MYSQL_PORT
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        print(f"[OK] Database `{MYSQL_DB}` verified/created!")
        cursor.close()
        conn.close()

        # Step 2: Connect to serisense_db & create users table
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            port=MYSQL_PORT
        )
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                uid VARCHAR(255) UNIQUE,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash TEXT,
                full_name VARCHAR(255),
                phone VARCHAR(50),
                state VARCHAR(100),
                district VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ''')
        print("[OK] Table `users` created successfully in MySQL!")

        # Step 3: Insert seed demo users
        demo_users = [
            ('user_001', 'farmer.pavir@gmail.com', generate_password_hash('Farmer@123'), 'Pavithran K', '+91 9876543210', 'Tamil Nadu', 'Erode'),
            ('user_002', 'ramesh.seri@gmail.com', generate_password_hash('RameshPass'), 'Ramesh Kumar', '+91 9123456789', 'Karnataka', 'Ramanagara'),
            ('user_003', 'anita.silk@gmail.com', generate_password_hash('Anita@2026'), 'Anita Devi', '+91 9988776655', 'Andhra Pradesh', 'Anantapur')
        ]

        for uid, email, p_hash, name, phone, state, district in demo_users:
            cursor.execute('''
                INSERT INTO users (uid, email, password_hash, full_name, phone, state, district)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    full_name = VALUES(full_name),
                    phone = VALUES(phone),
                    state = VALUES(state),
                    district = VALUES(district);
            ''', (uid, email, p_hash, name, phone, state, district))

        conn.commit()
        cursor.close()
        conn.close()

        print("\n=======================================================")
        print("🎉 SUCCESS! MySQL Database setup complete.")
        print("You can now open MySQL Workbench, connect to localhost:3306,")
        print(f"open database `{MYSQL_DB}` and inspect table `users`!")
        print("=======================================================\n")

    except Exception as e:
        print(f"\n[ERROR] Could not connect to MySQL: {e}")
        print("Please verify that your local MySQL service is running and password is correct.\n")

if __name__ == '__main__':
    init_mysql()
