"""
Feature layer: raw 1-minute candles + article scores -> per-coin bar features
in the `features_bar` table.

`features_bar` is the CONTRACT between the data pipeline and everything
downstream (signals, backtesting, API, dashboard). Downstream code reads only
this table, never series_builder or the raw tables, so the engine that fills it
(pandas today, PySpark later) can be swapped without touching consumers.
"""