"""Validation gate: promote the latest registered model version to Production
only if its `val_auc` is within `auc_tolerance` of the current Production.

If no Production version exists yet, the latest is promoted unconditionally
(bootstrap case).
"""

from __future__ import annotations

from typing import Any


def _read_metric(client: Any, run_id: str, key: str) -> float | None:
    """Fetch a metric from an MLFlow run. Returns None if missing."""
    try:
        run = client.get_run(run_id)
    except Exception:
        return None
    val = run.data.metrics.get(key)
    return float(val) if val is not None else None


def validate_and_promote(
    model_name: str,
    mlflow_uri: str,
    auc_tolerance: float = 0.005,
    metric_key: str = "val_auc",
) -> dict[str, Any]:
    """Decide whether to promote the newest version of `model_name`.

    Returns a dict describing the decision: whether the new version was
    promoted, its metric value, and the prior Production metric (if any).
    """
    try:
        from mlflow.tracking import MlflowClient  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "validate_and_promote requires mlflow. pip install mlflow."
        ) from e

    client = MlflowClient(tracking_uri=mlflow_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return {
            "promoted": False,
            "reason": "no_versions",
            "model_name": model_name,
        }

    latest = max(versions, key=lambda v: int(v.version))
    new_metric = _read_metric(client, latest.run_id, metric_key)

    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if not prod_versions:
        client.transition_model_version_stage(
            model_name, latest.version, "Production"
        )
        return {
            "promoted": True,
            "reason": "bootstrap",
            "version": latest.version,
            "new_metric": new_metric,
        }

    prod = prod_versions[0]
    if prod.version == latest.version:
        return {
            "promoted": False,
            "reason": "already_production",
            "version": latest.version,
            "new_metric": new_metric,
        }
    prod_metric = _read_metric(client, prod.run_id, metric_key)

    if new_metric is None:
        return {
            "promoted": False,
            "reason": "new_metric_missing",
            "version": latest.version,
        }
    if prod_metric is None:
        # Treat as bootstrap-equivalent: promote and archive the malformed prior.
        client.transition_model_version_stage(
            model_name, latest.version, "Production",
            archive_existing_versions=True,
        )
        return {
            "promoted": True,
            "reason": "prior_prod_missing_metric",
            "version": latest.version,
            "new_metric": new_metric,
        }

    if new_metric >= prod_metric - auc_tolerance:
        client.transition_model_version_stage(
            model_name, latest.version, "Production",
            archive_existing_versions=True,
        )
        return {
            "promoted": True,
            "reason": "metric_within_tolerance",
            "version": latest.version,
            "new_metric": new_metric,
            "prod_metric": prod_metric,
            "tolerance": auc_tolerance,
        }

    return {
        "promoted": False,
        "reason": "metric_regression",
        "version": latest.version,
        "new_metric": new_metric,
        "prod_metric": prod_metric,
        "tolerance": auc_tolerance,
    }
