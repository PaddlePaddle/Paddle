from __future__ import annotations

"""Global test configuration for paddle.metric."""
import os

import pytest

import paddle

NUM_PROCESSES = 2
NUM_BATCHES = 2 * NUM_PROCESSES
BATCH_SIZE = 32
NUM_CLASSES = 5
EXTRA_DIM = 3
THRESHOLD = 0.5
MAX_PORT = 8100
START_PORT = 8088
CURRENT_PORT = START_PORT
USE_PYTEST_POOL = os.getenv("USE_PYTEST_POOL", "0") == "1"


@pytest.fixture
def use_deterministic_algorithms():
    """Set deterministic algorithms for the test."""
    paddle.use_deterministic_algorithms(True)
    yield
    paddle.use_deterministic_algorithms(False)


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize distributed environment for testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(START_PORT)
    if not paddle.distributed.is_initialized():
        paddle.distributed.init_parallel_env()
