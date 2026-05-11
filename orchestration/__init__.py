"""Prefect-based daily retrain orchestration.

Hard dependencies (Prefect, MLFlow) are imported lazily inside individual
modules so the wider trainer package remains importable without them.
"""
