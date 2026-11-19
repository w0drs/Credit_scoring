"""Enforce the lab's additive SQLite migration policy without executing SQL."""
from pathlib import Path
import subprocess

from sqlglot import exp, parse


def main():
    files = subprocess.check_output(["git", "ls-files", "-z", "--", "*.sql"]).decode()
    paths = [Path(name) for name in files.split("\0") if name]
    errors = []
    for path in paths:
        statements = parse(path.read_text(), read="sqlite")
        for number, statement in enumerate(filter(None, statements), 1):
            if isinstance(statement, exp.Create) and statement.kind in {"TABLE", "INDEX"}:
                allowed = not statement.args.get("replace") and statement.expression is None
            elif isinstance(statement, exp.Alter) and statement.kind == "TABLE":
                actions = statement.args["actions"]
                allowed = bool(actions) and all(isinstance(action, exp.ColumnDef) for action in actions)
            else:
                allowed = False
            if not allowed:
                errors.append(f"{path}:{number}: {type(statement).__name__} not allowed")
    for error in errors:
        print(error)
    print(f"SQL policy: {len(paths)} tracked file(s), {len(errors)} violation(s)")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
