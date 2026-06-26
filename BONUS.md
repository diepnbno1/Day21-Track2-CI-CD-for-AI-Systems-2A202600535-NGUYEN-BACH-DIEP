# Bonus implementation notes

This repository implements the advanced challenges as follows.

## Bonus 1: Remote MLflow tracking with DagsHub

The workflow supports DagsHub remote tracking through optional GitHub Actions
secrets:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

When `MLFLOW_TRACKING_URI` is present, `.github/workflows/mlops.yml` exports the
MLflow environment variables before `python src/train.py`. When the secrets are
not configured, training falls back to local `sqlite:///mlflow.db`.

DagsHub setup still has to be done in the DagsHub web UI:

1. Create or log in to a DagsHub account.
2. Connect/import this GitHub repository.
3. Copy the repo MLflow tracking URL, usually:
   `https://dagshub.com/<dagshub-user>/<dagshub-repo>.mlflow`.
4. Create a DagsHub access token.
5. Add the three secrets above in GitHub repository settings.

## Bonus 2: Multiple algorithms

`src/train.py` supports these `model_type` values from `params.yaml`:

- `random_forest`
- `gradient_boosting`
- `logistic_regression`

Each algorithm has its own parameter block in `params.yaml`.

## Bonus 3: Automatic performance report

`src/train.py` writes `outputs/report.txt` after training. The report contains:

- Confusion matrix
- Per-class precision
- Per-class recall
- Per-class F1 score and support

GitHub Actions uploads both `outputs/metrics.json` and `outputs/report.txt` as
the `metrics-and-report` artifact.

## Bonus 4: Rollback / safe deployment gate

The train job uploads candidate artifacts to:

`s3://<bucket>/models/candidates/<commit-sha>/`

The eval job compares the new accuracy with `models/latest/metrics.json`. It
only promotes the candidate to `models/latest/` when:

- accuracy is at least `0.70`
- accuracy is greater than or equal to the currently deployed model accuracy

If the new model is worse, deployment is cancelled and the previous
`models/latest/model.pkl` remains active.

## Bonus 5: Data drift warning

`src/train.py` computes the training label distribution for classes `0`, `1`,
and `2`. If any class is below 10% of the training set, the warning is printed
to CI logs.

The label distribution and warnings are also written to `outputs/metrics.json`.
