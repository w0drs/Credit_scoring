#!/usr/bin/env bash
set -euo pipefail
image="${LAB8_IMAGE:?}"
state="${LAB8_STATE:?}"
stand="${2:-}"
db="$state/$stand.db"
fixture=data/database/applications.db
mkdir -p "$state"

schema() { python3 ci/schema_snapshot.py "$1"; }
yoyo() {
    docker run --rm --user "$(id -u):$(id -g)" -e LOGNAME=scoring-ci \
      -v "$state:/state" "$image" yoyo "$@"
}
total() { sqlite3 -readonly -batch -noheader "$1" 'SELECT count(*) FROM applications_database;'; }
integrity() { test "$(sqlite3 -readonly -batch -noheader "$1" 'PRAGMA integrity_check;')" = ok; }
copy_db() {
    test ! -e "$2"
    sqlite3 -readonly -batch -bail "$1" ".backup '$2'"
}

case "$1" in
    seed)
        test ! -e "$db"
        cp "$fixture" "$db"
        echo "$stand seeded: $(total "$db") rows"
        ;;
    first-test)
        yoyo apply --batch --no-config-file --database sqlite:////state/test.db -r 001_review db/migrations
        schema "$state/test.db" | diff -u db/schema/review.sql -
        echo 'TEST: migration 001_review applied; schema matches Git'
        ;;
    manual-stage)
        copy_db "$state/stage.db" "$state/stage-before.db"
        echo 'Operator-selected SQL: sqlite3 STAGE.db ".read db/migrations/001_review.sql"'
        sqlite3 -batch -bail "$state/stage.db" '.read db/migrations/001_review.sql'
        schema "$state/stage.db" | diff -u db/schema/review.sql -
        yoyo mark --batch --no-config-file --database sqlite:////state/stage.db -r 001_review db/migrations
        echo 'STAGE: SQL applied using sqlite3 .read; recorded 001_review without rerunning it'
        ;;
    verify)
        schema "$db" | diff -u "$3" -
        integrity "$db"
        test "$(total "$db")" = "$(total "$fixture")"
        echo "$stand: schema and integrity OK; $(total "$db") rows"
        ;;
    test-latest)
        yoyo apply --batch --no-config-file --database sqlite:////state/test.db db/migrations
        schema "$state/test.db" | diff -u db/schema/current.sql -
        echo 'TEST: all migrations applied; schema matches db/schema/current.sql'
        ;;
    backup-prod)
        umask 077
        copy_db "$state/prod.db" "$state/prod-before.db"
        integrity "$state/prod-before.db"
        diff -u <(schema "$state/prod.db") <(schema "$state/prod-before.db")
        test "$(total "$state/prod.db")" = "$(total "$state/prod-before.db")"
        echo "PROD backup verified before migration: prod-before.db; $(total "$state/prod-before.db") rows"
        ;;
    promote)
        from="$3"
        source="$state/$from.db"
        schema "$source" > "$state/source.sql"
        schema "$db" > "$state/target.sql"
        diff -u db/schema/current.sql "$state/source.sql"
        if cmp -s "$state/source.sql" "$state/target.sql"; then
            echo "$from == $stand; skip migration"
            exit 0
        fi
        echo "$from != $stand before migration"
        diff -U0 --label "$stand" --label "$from" "$state/target.sql" "$state/source.sql" || test "$?" = 1
        if [[ "$stand" == prod ]]; then
            test -f "$state/prod-before.db"
            diff -u "$state/target.sql" <(schema "$state/prod-before.db")
        fi
        copy_db "$db" "$state/preview.db"
        yoyo apply --batch --no-config-file --database sqlite:////state/preview.db db/migrations
        schema "$state/preview.db" | diff -u "$state/source.sql" -
        echo 'Migration preview on a disposable copy matches the source'
        yoyo apply --batch --no-config-file --database "sqlite:////state/$stand.db" db/migrations
        schema "$db" | diff -u "$state/source.sql" -
        echo "$from == $stand after migration"
        ;;
    restore-prod)
        copy_db "$state/prod-before.db" "$state/prod-restored.db"
        integrity "$state/prod-restored.db"
        test "$(total "$state/prod-restored.db")" = "$(total "$fixture")"
        diff -u <(schema "$fixture") <(schema "$state/prod-restored.db")
        echo "PROD restore verified in a separate DB: pre-migration schema; $(total "$state/prod-restored.db") rows"
        sha256sum "$state/prod-before.db"
        ;;
    *) exit 2 ;;
esac
