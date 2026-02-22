import sqlite3
from typing import Optional, Dict, Any, List
from passlib.context import CryptContext
from datetime import datetime, timedelta

DB_NAME = "users.db"
# Use pbkdf2_sha256 to avoid bcrypt 72-byte limit/dependency issues on Windows
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create table with all required fields
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY, 
                    password TEXT, 
                    email TEXT, 
                    phone TEXT,
                    avatar TEXT,
                    preferred_output_format TEXT DEFAULT 'markdown',
                    failed_login_attempts INTEGER DEFAULT 0,
                    account_locked_until TIMESTAMP
                )''')
    
    # Simple migration: check if columns exist, if not add them
    c.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in c.fetchall()]
    
    if 'avatar' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
    if 'preferred_output_format' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN preferred_output_format TEXT DEFAULT 'markdown'")
    if 'failed_login_attempts' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0")
    if 'account_locked_until' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN account_locked_until TIMESTAMP")
    if 'persona' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN persona TEXT DEFAULT ''")
        
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def create_user(username, password, email=None, phone=None, avatar=None, preferred_output_format="markdown"):
    hashed_pwd = hash_password(password)
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email, phone, avatar, preferred_output_format) VALUES (?, ?, ?, ?, ?, ?)", 
                  (username, hashed_pwd, email, phone, avatar, preferred_output_format))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def update_user(username: str, data: Dict[str, Any]) -> bool:
    """
    Update user profile fields.
    """
    valid_fields = ['email', 'phone', 'avatar', 'preferred_output_format', 'password', 'persona']
    updates = []
    values = []
    
    for key, value in data.items():
        if key in valid_fields and value is not None:
            if key == 'password':
                value = hash_password(value)
            updates.append(f"{key} = ?")
            values.append(value)
            
    if not updates:
        return False
        
    values.append(username)
    query = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(query, values)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Update error: {e}")
        return False

def verify_login(username, password) -> Dict[str, Any]:
    """
    Verify login credentials and handle locking logic.
    Returns dict with 'success': bool, 'message': str
    """
    user = get_user(username)
    if not user:
        return {"success": False, "message": "User not found"}
        
    # Check if locked
    if user['account_locked_until']:
        locked_until = user['account_locked_until']
        # Handle timestamp string parsing if needed (SQLite stores as string)
        # Assuming it comes out as string "YYYY-MM-DD HH:MM:SS.ssssss"
        try:
            locked_until_dt = datetime.fromisoformat(locked_until)
            if datetime.now() < locked_until_dt:
                return {"success": False, "message": f"Account locked until {locked_until}"}
        except ValueError:
            pass # Invalid format, ignore or reset

    # Verify password
    # Note: Legacy plain text passwords support (for existing users from previous version)
    # If verify fails, check if it matches plain text. If so, update to hash.
    password_valid = False
    try:
        if verify_password(password, user['password']):
            password_valid = True
    except Exception:
        # Fallback for plain text
        if password == user['password']:
            password_valid = True
            # Update to hash for future
            update_user(username, {'password': password})

    conn = get_db_connection()
    c = conn.cursor()
    
    if password_valid:
        # Reset failures
        c.execute("UPDATE users SET failed_login_attempts = 0, account_locked_until = NULL WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Login successful"}
    else:
        # Increment failure
        new_failures = (user['failed_login_attempts'] or 0) + 1
        lock_msg = ""
        
        if new_failures >= 5:
            # Lock for 15 minutes
            lock_time = datetime.now() + timedelta(minutes=15)
            c.execute("UPDATE users SET failed_login_attempts = ?, account_locked_until = ? WHERE username = ?", 
                      (new_failures, lock_time, username))
            lock_msg = " Account locked for 15 minutes."
        else:
            c.execute("UPDATE users SET failed_login_attempts = ? WHERE username = ?", (new_failures, username))
            
        conn.commit()
        conn.close()
        return {"success": False, "message": f"Invalid password.{lock_msg}"}

def get_all_users() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT username, email, phone, avatar, preferred_output_format, failed_login_attempts, account_locked_until FROM users")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_user(username: str) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        deleted = c.rowcount > 0
        conn.close()
        return deleted
    except Exception as e:
        print(f"Delete error: {e}")
        conn.close()
        return False
