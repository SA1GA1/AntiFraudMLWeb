"""Register the daily flow as a Prefect deployment with a cron schedule.

Run once after Prefect is set up:

    python -m orchestration.deployment

Then start a worker on the host:

    prefect worker start --pool default

The cron + timezone are env-driven so deployments stay portable between
hosts. Defaults to 03:00 Europe/Moscow.
"""

from __future__ import annotations

import os
import sys


def _build() -> None:
    try:
        from prefect.client.schemas.schedules import CronSchedule  # type: ignore
    except ImportError:  # older prefect 2.x layout
        from prefect.server.schemas.schedules import CronSchedule  # type: ignore

    from .daily_flow import daily_flow

    cron = os.environ.get("FRAUD_CRON", "0 3 * * *")
    tz = os.environ.get("FRAUD_CRON_TZ", "Europe/Moscow")
    work_pool = os.environ.get("FRAUD_WORK_POOL", "default")

    daily_flow.deploy(
        name="nightly-retrain",
        work_pool_name=work_pool,
        schedules=[CronSchedule(cron=cron, timezone=tz)],
        tags=["fraud", "web", "daily"],
    )


if __name__ == "__main__":
    try:
        _build()
    except Exception as e:  # pragma: no cover — surfaces dep errors to user
        print(f"deployment failed: {e}", file=sys.stderr)
        raise
