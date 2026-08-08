"""
Sanity check for the logging setup. Run directly with:
    python tests/test_logging.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("This is a DEBUG message (won't show if log_level is INFO)")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")
logger.error("This is an ERROR message")

print("\nCheck logs/app.log — it should contain the INFO, WARNING, and ERROR lines above.")