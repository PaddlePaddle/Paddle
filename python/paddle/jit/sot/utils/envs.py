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

import os
from contextlib import contextmanager
from typing import Dict, List

import paddle
from paddle.utils.environments import (
    BooleanEnvironmentVariable,
    EnvironmentVariable,
    EnvironmentVariableGuard,
    IntegerEnvironmentVariable,
    StringEnvironmentVariable,
)


class PEP508LikeEnvironmentVariable(EnvironmentVariable[Dict[str, List[str]]]):
    """
    Environment variable parser following PEP 508 extras specification syntax.
    https://peps.python.org/pep-0508/

    Processes strings using PEP 508-style bracket notation for optional components:
    "feat1[opt1,opt2], feat2[opt3,opt4]" -> {'feat1': ['opt1', 'opt2'], 'feat2': ['opt3', 'opt4']}
    """

    def __init__(self, name: str, default: dict[str, list[str]]):
        super().__init__(name, default)
        assert isinstance(default, dict), "default must be a dict"

    def parse_from_string(self) -> dict[str, list[str]]:
        env_var = os.getenv(self.name)
        if env_var is None or env_var == "":
            return self.default
        items = self.split_by_unbracketed_commas(env_var)
        ret = {}
        for item in items:
            ret.update(self.parse_parameterized_key(item))
        return ret

    def convert_to_string(self, value: dict[str, list[str]]) -> str:
        assert isinstance(value, dict), "The input must be a dict"
        assert all(
            isinstance(x, str) for x in value.keys()
        ), "Keys must be a string"
        assert all(
            isinstance(x, list) for x in value.values()
        ), "Values must be a list"

        env_list = []
        for k, v in value.items():
            env_list.append(f"{k}" + (f"[{','.join(v)}]" if len(v) else ""))

        return ",".join(env_list)

    @staticmethod
    def split_by_unbracketed_commas(input_str: str) -> list[str]:
        """Split string by commas that are not enclosed in square brackets"""
        # "feat1[opt1,opt2], feat2[opt3], feat3" -> ["feat1[opt1,opt2]", "feat2[opt3]", "feat3"]
        bracket_depth = 0
        split_parts = []
        _start = 0

        for _current, char in enumerate(input_str):
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(
                    0, bracket_depth - 1
                )  # Prevent negative depth

            if char == "," and bracket_depth == 0:
                split_parts.append(input_str[_start:_current].strip())
                _start = _current + 1  # Skip comma

        # Add remaining content after last comma
        if remaining := input_str[_start:].strip():
            split_parts.append(remaining)

        return split_parts

    @staticmethod
    def parse_parameterized_key(input_str: str) -> dict[str, list[str]]:
        """Parse key with parameters in brackets into a dictionary."""

        start_bracket = input_str.find("[")
        end_bracket = input_str.rfind("]")

        if start_bracket == -1 or end_bracket == -1:
            return {input_str: []}

        parameter_key = input_str[:start_bracket].strip()

        # Extract and clean parameters
        parameters_str = input_str[start_bracket + 1 : end_bracket]
        parameter_values = [
            v.strip() for v in parameters_str.split(",") if v.strip()
        ]

        return {parameter_key: parameter_values}


ENV_MIN_GRAPH_SIZE = IntegerEnvironmentVariable("MIN_GRAPH_SIZE", 10)
ENV_SOT_LOG_LEVEL = IntegerEnvironmentVariable("SOT_LOG_LEVEL", 0)
ENV_STRICT_MODE = BooleanEnvironmentVariable("STRICT_MODE", False)
ENV_SOT_WITH_CONTROL_FLOW = BooleanEnvironmentVariable(
    "SOT_WITH_CONTROL_FLOW", True
)
ENV_SOT_EXPORT = StringEnvironmentVariable("SOT_EXPORT", "")
ENV_SOT_ALLOW_DYNAMIC_SHAPE = BooleanEnvironmentVariable(
    "SOT_ALLOW_DYNAMIC_SHAPE",
    # Enable SOT dynamic shape as default in PIR mode only
    paddle.framework.use_pir_api(),
)
ENV_SOT_ENABLE_FASTER_GUARD = BooleanEnvironmentVariable(
    "SOT_ENABLE_FASTER_GUARD",
    False,
)
ENV_SOT_ENABLE_GUARD_TREE = BooleanEnvironmentVariable(
    "SOT_ENABLE_GUARD_TREE",
    False,
)
ENV_SOT_EVENT_LEVEL = IntegerEnvironmentVariable("SOT_EVENT_LEVEL", 0)
ENV_ENABLE_SOT_STEP_PROFILER = BooleanEnvironmentVariable(
    "ENABLE_SOT_STEP_PROFILER", False
)
ENV_SOT_BREAK_GRAPH_ON_GET_SYMBOLIC_VALUE = BooleanEnvironmentVariable(
    "SOT_BREAK_GRAPH_ON_GET_SYMBOLIC_VALUE", False
)
ENV_SOT_COLLECT_INFO = PEP508LikeEnvironmentVariable("SOT_COLLECT_INFO", {})
ENV_SOT_SERIALIZE_INFO = BooleanEnvironmentVariable("SOT_SERIALIZE_INFO", False)
ENV_SOT_FORCE_FALLBACK_SIR_IDS = StringEnvironmentVariable(
    "SOT_FORCE_FALLBACK_SIR_IDS", ""
)


@contextmanager
def strict_mode_guard(value: bool):
    with EnvironmentVariableGuard(ENV_STRICT_MODE, value):
        yield


@contextmanager
def min_graph_size_guard(value: int):
    with EnvironmentVariableGuard(ENV_MIN_GRAPH_SIZE, value):
        yield


@contextmanager
def with_control_flow_guard(value: bool):
    with EnvironmentVariableGuard(ENV_SOT_WITH_CONTROL_FLOW, value):
        yield


@contextmanager
def export_guard(value: str):
    with EnvironmentVariableGuard(ENV_SOT_EXPORT, value):
        yield


@contextmanager
def allow_dynamic_shape_guard(value: bool):
    with EnvironmentVariableGuard(ENV_SOT_ALLOW_DYNAMIC_SHAPE, value):
        yield


@contextmanager
def faster_guard_guard(value: bool):
    with EnvironmentVariableGuard(ENV_SOT_ENABLE_FASTER_GUARD, value):
        yield


@contextmanager
def guard_tree_guard(value: bool):
    with EnvironmentVariableGuard(ENV_SOT_ENABLE_GUARD_TREE, value):
        yield


@contextmanager
def sot_step_profiler_guard(value: bool):
    with EnvironmentVariableGuard(ENV_ENABLE_SOT_STEP_PROFILER, value):
        yield
