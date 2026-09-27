"""
Supervisor for the storage consumers (market + sentiment).

Why separate PROCESSES instead of threads:
  - A crashed thread dies quietly while the others keep running ("silent
    partial death"). A supervisor can detect a dead process via its exit code
    and restart it.
  - daemon threads are killed abruptly at exit, skipping cleanup
    (consumer.close(), clean LeaveGroup).
  - On Windows, Thread.join() without a timeout can block Ctrl+C.

Policy: let-it-crash + restart.
  Each consumer fails fast (e.g. DB still down after its retries) and never
  commits an unwritten offset, so restarting it is always safe
  (at-least-once + idempotent inserts).

Crash-loop guard:
  If one consumer needs more than MAX_RESTARTS restarts within
  RESTART_WINDOW_SECONDS, something is persistently wrong: stop everything
  and exit non-zero, loudly, instead of restarting forever.

Run with: python -m streaming.run_storage_consumers
"""

import multiprocessing as mp
import sys
import time
from collections import deque

from config.logging_config import get_logger
from streaming import market_consumer, sentiment_consumer

logger = get_logger(__name__)

CHECK_INTERVAL_SECONDS = 2
RESTART_BACKOFF_BASE_SECONDS = 5      # 5, 10, 20, 40, 60, 60 ...
RESTART_BACKOFF_MAX_SECONDS = 60
MAX_RESTARTS = 5
RESTART_WINDOW_SECONDS = 600          # 10 minutes
SHUTDOWN_TIMEOUT_SECONDS = 15

# name -> importable top-level function (required for Windows 'spawn' start method)
WORKERS = {
    "market-consumer": market_consumer.run,
    "sentiment-consumer": sentiment_consumer.run,
}


class Worker:
    """One supervised consumer process plus its restart history."""

    def __init__(self, name: str, target) -> None:
        self.name = name
        self.target = target
        self.process = None
        self.restart_times = deque()   # monotonic timestamps of recent restarts
        self.next_start_at = 0.0

    def start(self) -> None:
        self.process = mp.Process(target=self.target, name=self.name)
        self.process.start()
        logger.info(f"Started {self.name} (pid={self.process.pid})")


def _supervise(workers: list) -> int:
    """Watch the workers forever; restart any that die. Returns an exit code
    only if the crash-loop guard trips."""
    while True:
        now = time.monotonic()
        for w in workers:
            # Waiting out a restart backoff?
            if w.process is None:
                if now >= w.next_start_at:
                    w.start()
                continue

            if w.process.is_alive():
                continue

            # The process has exited but we did not ask it to: unexpected.
            exit_code = w.process.exitcode
            w.process = None

            while w.restart_times and now - w.restart_times[0] > RESTART_WINDOW_SECONDS:
                w.restart_times.popleft()

            if len(w.restart_times) >= MAX_RESTARTS:
                logger.critical(
                    f"{w.name} crashed {MAX_RESTARTS} times within "
                    f"{RESTART_WINDOW_SECONDS // 60} min (last exitcode={exit_code}). "
                    f"Crash loop: stopping ALL consumers. Fix the root cause "
                    f"(is the DB/Kafka up?) and restart the supervisor."
                )
                return 1

            delay = min(RESTART_BACKOFF_BASE_SECONDS * 2 ** len(w.restart_times),
                        RESTART_BACKOFF_MAX_SECONDS)
            w.restart_times.append(now)
            w.next_start_at = now + delay
            logger.error(
                f"{w.name} exited unexpectedly (exitcode={exit_code}). Restarting in {delay}s "
                f"(restart {len(w.restart_times)}/{MAX_RESTARTS} in the last "
                f"{RESTART_WINDOW_SECONDS // 60} min)."
            )

        time.sleep(CHECK_INTERVAL_SECONDS)


def _shutdown(workers: list, graceful: bool) -> None:
    """graceful=True (Ctrl+C): children received Ctrl+C too and are closing
    themselves, so wait for them. graceful=False (crash loop): terminate the
    rest now. Either way is safe: offsets are only committed after writes."""
    for w in workers:
        if w.process is None:
            continue
        if graceful:
            w.process.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        if w.process.is_alive():
            if graceful:
                logger.warning(f"{w.name} did not stop within {SHUTDOWN_TIMEOUT_SECONDS}s; terminating.")
            w.process.terminate()
            w.process.join(timeout=5)
        logger.info(f"{w.name} stopped (exitcode={w.process.exitcode}).")


def main() -> int:
    workers = [Worker(name, target) for name, target in WORKERS.items()]
    for w in workers:
        w.start()
    logger.info("Storage consumers running under supervisor. Press Ctrl+C to stop.")

    exit_code = 0
    graceful = True
    try:
        exit_code = _supervise(workers)
        graceful = False           # only reached via the crash-loop guard
    except KeyboardInterrupt:
        logger.info("Ctrl+C received; waiting for consumers to close cleanly...")
    finally:
        _shutdown(workers, graceful=graceful)
        logger.info("Supervisor stopped.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())