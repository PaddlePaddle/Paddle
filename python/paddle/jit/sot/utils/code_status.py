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

import inspect
from enum import Enum

import paddle

from .utils import Singleton, log


class CodeState(Enum):
    UNKNOW = 1
    WITH_GRAPH = 2
    WITHOUT_GRAPH = 3


class CodeInfo:
    def __init__(self):
        self.state = CodeState.UNKNOW
        self.counter = 0

    def __repr__(self):
        return f"state: {self.state}, counter: {self.counter}"


@Singleton
class CodeStatus:
    WITH_GRAPH_API = [
        paddle.nn.Layer.__call__.__code__,
        paddle.nn.Layer._dygraph_call_func.__code__,
    ]

    def __init__(self):
        self.code_map = {}
        self.setup_code_map()

    def setup_code_map(self):
        for code in self.WITH_GRAPH_API:
            info = CodeInfo()
            info.state = CodeState.WITH_GRAPH
            self.code_map[code] = info

    def clear(self):
        self.code_map.clear()
        self.setup_code_map()

    def is_code_without_graph(self, code):
        if code not in self.code_map:
            info = CodeInfo()
            self.code_map[code] = info
        else:
            info = self.code_map[code]

        if info.state == CodeState.WITHOUT_GRAPH:
            return True
        if info.state == CodeState.UNKNOW:
            info.counter += 1
            if info.counter >= 10:
                log(
                    3,
                    f"[CodeStatus] Switch state to WITHOUT_GRAPH for {code}\n",
                )
                info.state = CodeState.WITHOUT_GRAPH
        return False

    def trace_back_frames(self):
        frame = inspect.currentframe()
        while frame.f_back is not None:
            frame = frame.f_back
            code = frame.f_code
            if code in self.code_map:
                info = self.code_map[code]
                if info.state != CodeState.WITH_GRAPH:
                    log(
                        3,
                        f"[CodeStatus] Switch state to WITH_GRAPH for {code}\n",
                    )
                    info.state = CodeState.WITH_GRAPH
