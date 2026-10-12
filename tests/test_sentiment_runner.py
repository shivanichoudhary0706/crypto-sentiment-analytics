"""Tests for the sentiment ingestion runner's --source selection."""

import asyncio

import pytest

from ingestion.sentiment import run_sentiment_ingestion as runner


def test_resolve_all_returns_every_poller():
    assert runner.resolve_sources("all") == ["rss", "gdelt"]


@pytest.mark.parametrize("source", ["rss", "gdelt"])
def test_resolve_single_source(source):
    assert runner.resolve_sources(source) == [source]


def test_resolve_unknown_source_raises():
    with pytest.raises(ValueError):
        runner.resolve_sources("reddit")


def test_parse_args_default_is_all():
    assert runner.parse_args([]).source == "all"


def test_parse_args_rejects_unknown_source():
    with pytest.raises(SystemExit):
        runner.parse_args(["--source", "twitter"])


@pytest.mark.parametrize(
    "source, expected",
    [("rss", ["rss"]), ("gdelt", ["gdelt"]), ("all", ["rss", "gdelt"])],
)
def test_run_starts_only_selected_pollers(monkeypatch, source, expected):
    called = []

    def make_fake(name):
        async def fake():
            called.append(name)
        return fake

    monkeypatch.setattr(
        runner, "POLLERS", {"rss": make_fake("rss"), "gdelt": make_fake("gdelt")}
    )
    asyncio.run(runner.run(source))
    assert sorted(called) == sorted(expected)