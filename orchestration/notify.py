"""Notify AntiFraudMain to hot-reload its model via the registry.

Backend exposes ``POST /admin/reload-model`` (see AntiFraudMain side of the
plan). This module is the daily flow's final step — fires after a successful
``promote`` so the inference service picks up the new Production version
without a restart.

Kept stdlib-only so it can run inside Prefect tasks without extra deps.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def notify_backend(
    url: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    retries: int = 3,
    timeout: float = 30.0,
) -> dict[str, Any]:
    body = json.dumps(payload or {}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8") or "{}"
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"raw": raw, "status": resp.status}
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"notify_backend failed after {retries} attempts: {last_err}")
