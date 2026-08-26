CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================
-- Market data (1-minute candles, closed candles only — see market_consumer.py)
-- ============================
CREATE TABLE IF NOT EXISTS market_data (
    time             TIMESTAMPTZ       NOT NULL,   -- = kline_start_time, hypertable partition column
    symbol           TEXT              NOT NULL,
    exchange         TEXT              NOT NULL,
    open             DOUBLE PRECISION  NOT NULL,
    high             DOUBLE PRECISION  NOT NULL,
    low              DOUBLE PRECISION  NOT NULL,
    close            DOUBLE PRECISION  NOT NULL,
    volume           DOUBLE PRECISION  NOT NULL,
    kline_close_time TIMESTAMPTZ       NOT NULL,
    ingested_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, time)
);

SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time
    ON market_data (symbol, time DESC);


-- ============================
-- Raw sentiment (news RSS + GDELT articles)
-- ============================
CREATE TABLE IF NOT EXISTS raw_sentiment (
    time         TIMESTAMPTZ NOT NULL,              -- = published_at, hypertable partition column
    source_type  TEXT        NOT NULL,
    source_name  TEXT        NOT NULL,
    title        TEXT        NOT NULL,
    summary      TEXT,
    url          TEXT        NOT NULL,
    coin_tags    TEXT[]      NOT NULL DEFAULT '{}',
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- TimescaleDB requires unique constraints on a hypertable to include the
    -- partition column, so this is (time, url) rather than a plain UNIQUE(url).
    UNIQUE (time, url)
);

SELECT create_hypertable('raw_sentiment', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_raw_sentiment_coin_tags
    ON raw_sentiment USING GIN (coin_tags);

CREATE INDEX IF NOT EXISTS idx_raw_sentiment_source_time
    ON raw_sentiment (source_name, time DESC);

-- ============================
-- Sentiment scores (FinBERT + VADER, one row per article-coin pair)
-- ============================
CREATE TABLE IF NOT EXISTS sentiment_scores (
    time              TIMESTAMPTZ       NOT NULL,   -- = raw_sentiment.time, hypertable partition column
    url               TEXT              NOT NULL,
    coin              TEXT              NOT NULL,
    source_name       TEXT              NOT NULL,
    finbert_label     TEXT              NOT NULL,   -- 'positive' | 'negative' | 'neutral'
    finbert_positive  DOUBLE PRECISION  NOT NULL,
    finbert_negative  DOUBLE PRECISION  NOT NULL,
    finbert_neutral   DOUBLE PRECISION  NOT NULL,
    finbert_compound  DOUBLE PRECISION  NOT NULL,   -- positive - negative, single scalar for lag detection
    vader_compound    DOUBLE PRECISION  NOT NULL,   -- VADER's built-in compound score, range -1 to 1
    scored_at         TIMESTAMPTZ       NOT NULL DEFAULT now(),
    UNIQUE (time, url, coin)
);

SELECT create_hypertable('sentiment_scores', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_sentiment_scores_coin_time
    ON sentiment_scores (coin, time DESC); 