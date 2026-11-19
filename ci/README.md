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

## Lab 8 rollout on GitHub-hosted runners

Dispatch `lab8-manual.yml` with migration `001_review` on the lab branch.
It tests that version in TEST, applies the chosen SQL explicitly in STAGE
and saves the resulting two database files without deploying STAGE. The manual
workflow must be present on the default branch to be dispatchable.

A push to `devops/lab8-github` starts `lab8-stage.yml`. It loads the last
successful manual checkpoint from that branch, updates TEST, compares the
SQLite schemas, applies the checked-in migrations to STAGE and saves the
verified image and database checkpoint. Create and push a tag named
`scoring-lab8-prod-v*` on the same verified commit to run `lab8-prod.yml`.
It requires the exact STAGE run, backs up an independent PROD database before
migrating, restores the backup into a separate database and publishes the
image it actually deployed. Checkpoints expire after two days; the backup
artifact expires after three.

These stand names identify isolated, short-lived lab environments. Their
containers are removed after the smoke checks. Only the test fixture already
in the repository is copied; never substitute live data for it.

For a server deployment, point `SCORING_DB` at a persistent SQLite file and
make a verified backup before running the SQL migrations in `db/migrations`
with yoyo. Use the approved image digest, supply `BACKEND_URL` to the
Streamlit process, and check the API after starting the release. Deployment
hosts and secrets must be configured separately before enabling a server
promotion workflow.
