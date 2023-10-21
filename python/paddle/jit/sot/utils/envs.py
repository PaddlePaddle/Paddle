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
