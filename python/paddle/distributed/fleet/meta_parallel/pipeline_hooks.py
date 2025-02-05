# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from collections import defaultdict
from typing import Callable


class PipelineHook:
    def __init__(self):
        self.hooks: dict[int, list[Callable]] = defaultdict(list)
        self._hooks_capacity = 0
        self.reset_current_id()

    def reset_current_id(self):
        self._current_id = 0

    def set_hooks_capacity(self, capacity: int):
        self._hooks_capacity = capacity

    def register_hook(self, hook_id: int, hook: Callable):
        assert (
            hook_id < self._hooks_capacity
        ), f"hook_id {hook_id} is out of range, maximum capacity is {self._hooks_capacity}."
        self.hooks[hook_id].append(hook)

    def run_hook(self):
        assert (
            self._current_id < self._hooks_capacity
        ), f"hook_id {self._current_id} is out of range, maximum capacity is {self._hooks_capacity}."
        for hook in self.hooks[self._current_id]:
            hook(self._current_id)
        self._current_id += 1

    @property
    def current_id(self):
        return self._current_id

    @property
    def hooks_capacity(self):
        return self._hooks_capacity
