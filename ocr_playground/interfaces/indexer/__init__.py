"""The corpus indexer: CocoIndex's entry point, and the projections it declares.

Deliberately free of imports. `config.py` and `projection.py` are plain stdlib and are what the tests
import; `flow.py` imports CocoIndex, which is optional. Re-exporting `build_app` here would make
`import ocr_playground.interfaces.indexer.config` fail on a machine that installed only
`requirements-dev.txt`. `main.py` imports `flow` directly, so a missing CocoIndex still fails loudly —
at the one call site that actually needs it.
"""
