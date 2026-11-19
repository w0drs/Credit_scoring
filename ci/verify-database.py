"""Rehearse TEST -> STAGE -> PROD on temporary copies, never on live databases."""
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from schema_snapshot import snapshot
from yoyo import get_backend, read_migrations

root = Path(__file__).resolve().parents[1]
state = Path(sys.argv[1]).resolve()
source = root / "data/database/applications.db"
expected = (root / "db/schema/current.sql").read_text()
migrations = read_migrations(str(root / "db/migrations"))
rows_query = "SELECT count(*) FROM applications_database"


def open_db(path, mode="ro"):
    return closing(sqlite3.connect(path.as_uri() + "?mode=" + mode, uri=True))


def migrate(path):
    backend = get_backend(f"sqlite:///{path}")
    with backend.lock():
        backend.apply_migrations(backend.to_apply(migrations))


def backup(source, target):
    with open_db(source) as src, open_db(target, "rwc") as dst:
        src.backup(dst)


def validate(database, rows, expected_schema):
    with open_db(database) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute(rows_query).fetchone() == rows
    assert snapshot(database) == expected_schema


def main():
    os.umask(0o077)
    databases = [state / f"{stand}.db" for stand in ("test", "stage", "prod")]
    for db in databases:
        if db.exists():
            raise FileExistsError(db)
        shutil.copyfile(source, db)
    test, stage, prod = databases
    with open_db(source) as db:
        rows = db.execute(rows_query).fetchone()
    baseline = snapshot(source)

    migrate(test)
    validate(test, rows, expected)
    assert snapshot(stage) != snapshot(test)
    migrate(stage)
    validate(stage, rows, snapshot(test))
    print(f"TEST -> STAGE: identical schemas; {rows[0]} records")

    copy = state / "prod-before.db"
    restore = state / "prod-restored.db"
    backup(prod, copy)
    validate(copy, rows, baseline)
    assert snapshot(prod) != snapshot(stage)
    migrate(prod)
    validate(prod, rows, snapshot(stage))
    migrate(prod)
    validate(prod, rows, expected)
    with open_db(prod) as db:
        assert db.execute(
            "SELECT review_status FROM applications_database WHERE id = ?", (123,)
        ).fetchone() == ("pending",)
    backup(copy, restore)
    validate(restore, rows, baseline)
    print(f"STAGE -> PROD: identical schemas; {rows[0]} records")
    print("Pre-migration backup restored to a separate DB; integrity OK")


if __name__ == "__main__":
    main()
