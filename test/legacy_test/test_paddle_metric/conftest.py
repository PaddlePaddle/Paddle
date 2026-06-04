# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
