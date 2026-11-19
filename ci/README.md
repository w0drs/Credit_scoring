# Scoring delivery

GitHub Actions checks tracked SQL on every branch and pull request. It builds the
scoring image, runs the Python tests, rehearses both migrations and a backup
restore on disposable databases, and checks the prediction and review APIs
against the migrated database.

A push to the default branch (`master`) or of a release tag matching
`scoring-v*` publishes the exact tested image to
`ghcr.io/<owner>/credit_scoring:<commit-sha>`. Pull requests and ordinary
development branches cannot publish. Copying and restoring the databases in CI
is a rehearsal, not a deployment to a persistent environment.

For a server deployment, point `SCORING_DB` at a persistent SQLite file and
make a verified backup before running the SQL migrations in `db/migrations`
with yoyo. Use the approved image digest, supply `BACKEND_URL` to the
Streamlit process, and check the API after starting the release. Deployment
hosts and secrets must be configured separately before enabling a server
promotion workflow.
