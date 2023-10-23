# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from contextlib import contextmanager

from paddle.utils.environments import (
    BooleanEnvironmentVariable,
    EnvironmentVariableGuard,
    IntegerEnvironmentVariable,
    StringEnvironmentVariable,
)

ENV_COST_MODEL = BooleanEnvironmentVariable("COST_MODEL", False)
ENV_MIN_GRAPH_SIZE = IntegerEnvironmentVariable("MIN_GRAPH_SIZE", 10)
ENV_SOT_LOG_LEVEL = IntegerEnvironmentVariable("SOT_LOG_LEVEL", 0)
ENV_STRICT_MODE = BooleanEnvironmentVariable("STRICT_MODE", False)
ENV_SHOW_TRACKERS = StringEnvironmentVariable("SHOW_TRACKERS", "")
ENV_CLEAN_CODE = BooleanEnvironmentVariable("CLEAN_CODE", False)


@contextmanager
def cost_model_guard(value: bool):
    with EnvironmentVariableGuard(ENV_COST_MODEL, value):
        yield


@contextmanager
def strict_mode_guard(value: bool):
    with EnvironmentVariableGuard(ENV_STRICT_MODE, value):
        yield
