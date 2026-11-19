"""Export application DDL using SQLite's native sqlite_schema catalog."""
import sqlite3
import sys
from contextlib import closing
from pathlib import Path


def snapshot(database):
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_schema WHERE sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name"
        ).fetchall()
    definitions = [
        sql.strip().rstrip(";") for name, sql in rows
        if name != "yoyo_lock" and not name.startswith(("sqlite_", "_yoyo_"))
    ]
    return "".join("\n".join(line.rstrip() for line in sql.splitlines()) + ";\n"
                   for sql in definitions)


if __name__ == "__main__":
    print(snapshot(sys.argv[1]), end="")
