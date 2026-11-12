
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


def get_database_path():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir / "feedback.db"


def initialize_database():
    database_path = get_database_path()

    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                correct_label TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_feedback(text, predicted_label, correct_label):
    initialize_database()

    database_path = get_database_path()

    is_correct = int(predicted_label == correct_label)

    created_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            INSERT INTO feedback (
                text,
                predicted_label,
                correct_label,
                is_correct,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            text,
            str(predicted_label),
            str(correct_label),
            is_correct,
            created_at
        ))


def get_pending_feedback():
    initialize_database()

    database_path = get_database_path()

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT id, text, correct_label
            FROM feedback
            WHERE LOWER(TRIM(correct_label)) IN ('positive', 'negative')
              AND TRIM(text) != ''
            ORDER BY id
        """).fetchall()

        return [dict(row) for row in rows]


def clear_feedback():
    initialize_database()

    database_path = get_database_path()

    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM feedback")
