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


class BubbleHook:
    def __init__(self):
        self.hooks: dict[int, list[Callable]] = defaultdict(list)

    def set_bubble_times(self, bubble_times: int):
        self.bubble_times = bubble_times

    def register_hook(self, bubble_id: int, hook: Callable):
        self.hooks[bubble_id].append(hook)

    def run_hook(self, bubble_id: int):
        if bubble_id not in self.hooks:
            return

        for hook in self.hooks[bubble_id]:
            hook(bubble_id)
