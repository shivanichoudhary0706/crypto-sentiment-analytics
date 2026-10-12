"""
Batch feature job: 1-minute candles -> OHLCV bars, plus sentiment aggregated per bar.

Run inside the Spark container:
  /opt/spark/bin/spark-submit /app/batch/jobs/build_features.py --lake /data/lake --bar 4h
"""
import argparse

from pyspark.sql import SparkSession, functions as F

BAR_TO_SPARK = {"1h": "1 hour", "4h": "4 hours"}


def build_bars(spark, lake: str, bar: str):
    md = spark.read.parquet(f"{lake}/market_data")
    return (md.groupBy("symbol", F.window("time", BAR_TO_SPARK[bar]).alias("w"))
              .agg(F.min_by("open", "time").alias("open"),     # first candle's open
                   F.max("high").alias("high"),
                   F.min("low").alias("low"),
                   F.max_by("close", "time").alias("close"),   # last candle's close
                   F.sum("volume").alias("volume"),
                   F.count(F.lit(1)).alias("n_candles"))
              .select("symbol", F.col("w.start").alias("bar_start"),
                      "open", "high", "low", "close", "volume", "n_candles"))


def build_sentiment(spark, lake: str, bar: str):
    ss = spark.read.parquet(f"{lake}/sentiment_scores")
    return (ss.groupBy("coin", F.window("time", BAR_TO_SPARK[bar]).alias("w"))
              .agg(F.avg("finbert_compound").alias("finbert_mean"),
                   F.avg("vader_compound").alias("vader_mean"),
                   F.count(F.lit(1)).alias("n_articles"))
              .select("coin", F.col("w.start").alias("bar_start"),
                      "finbert_mean", "vader_mean", "n_articles"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default="/data/lake")
    ap.add_argument("--bar", choices=sorted(BAR_TO_SPARK), default="4h")
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName(f"build_features_{args.bar}")
             # Without this, bar boundaries follow the container's local
             # timezone. Every timestamp in this project is UTC.
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())

    out = f"{args.lake}/features"
    bars = build_bars(spark, args.lake, args.bar)
    bars.write.mode("overwrite").partitionBy("symbol").parquet(f"{out}/bars_{args.bar}")

    sent = build_sentiment(spark, args.lake, args.bar)
    (sent.withColumn("coin_key", F.regexp_replace("coin", "/", "-"))
         .write.mode("overwrite").partitionBy("coin_key")
         .parquet(f"{out}/sentiment_{args.bar}"))

    print(f"bars_{args.bar}: {bars.count()} rows | sentiment_{args.bar}: {sent.count()} rows")
    spark.stop()


if __name__ == "__main__":
    main()