"""RAG system with BM25-only contextual retrieval.

This module configures logging for the entire application on import.
"""

import logging
import os
import warnings

# 🔇 CENTRALIZED LOGGING CONFIGURATION 🔇
# This runs when the package is first imported, silencing all the noisy libraries!

# 1. 🤫 gRPC silence (CRITICAL: Set BEFORE importing gRPC-based libraries like IBM Watsonx!)
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
os.environ["GRPC_TRACE_FUZZER"] = ""
os.environ["GLOG_minloglevel"] = "3"  # 3 = FATAL only
os.environ["GLOG_logtostderr"] = "0"  # Don't log to stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 2. 🤐 Suppress warnings from all libraries
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

# 3. 🔕 Set all noisy loggers to CRITICAL/ERROR
_noisy_loggers = [
    "google",
    "google.api_core",
    "google.auth",
    "grpc",
    "grpc._channel",
    "bm25s",
    "bm25s.hf",
    "transformers",
    "transformers.tokenization_utils_base",
]

for logger_name in _noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# 4. ✨ Initialize our Rich logging (WARNING level)
from hackathon.utils.logging import setup_logging  # noqa: E402

setup_logging()


def main() -> None:
    """Example entry point."""
    print("Hello from hackathon!")
