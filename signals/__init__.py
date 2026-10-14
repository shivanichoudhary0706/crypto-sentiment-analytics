"""
Signal layer: features_bar -> ML model -> BUY / HOLD / SELL per target coin,
written to the `signals` table that the API and dashboard read.

Reads ONLY features_bar (never the raw tables), so it is independent of how
features are computed.
"""