"""Analysis pipeline implementing ``data_analysis_procedures.md``.

Consumes the instrument's trial-level dataset (and the service-uptime event log)
and produces the descriptive, inferential, reliability, survival, and power
outputs each research question requires. Every procedure is paired with its
predetermined assumption checks and nonparametric fallbacks, and the whole
pipeline degrades gracefully on empty/zero-variance/all-censored inputs.
"""
