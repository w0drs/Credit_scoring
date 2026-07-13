"""
ml-retraining-pipeline
======================
Production ML retraining pipeline with drift-aware triggers,
champion/challenger evaluation, and automatic rollback.

Modules
-------
triggers   — PSI/KS/performance/time-based retraining triggers
data       — data collection and validation (Stage 1)
features   — preprocessor fit-and-save, training-serving skew guard (Stage 2)
train      — multi-candidate training with experiment tracking (Stage 3)
evaluation — champion/challenger evaluation gate (Stage 4)
deploy     — artifact promotion and rollback (Stage 5)
orchestrator — single entry point wiring all five stages
"""

__version__ = "1.0.0"
