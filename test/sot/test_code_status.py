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

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit import sot
from paddle.jit.sot.opcode_translator.skip_files import skip_function
from paddle.jit.sot.utils import strict_mode_guard
from paddle.jit.sot.utils.code_status import CodeState, CodeStatus


class SimpleNet1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layers = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(30)]
        )

    def forward(self, x):
        for i in range(len(self.layers)):
            sot.psdb.breakgraph()
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
        return x


class SimpleNet2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layers = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(30)]
        )

    def forward(self, x):
        sot.psdb.fallback()
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
        return x


def run_net(net, x):
    for i in range(20):
        x = net(x)
    return x


class TestCodeInfo(TestCaseBase):
    def test_case_1(self):
        CodeStatus().clear()
        net = SimpleNet1()
        inp = paddle.rand((10, 10))
        self.assert_results(run_net, net, inp)
        code_map = CodeStatus().code_map
        states = []
        for k, v in code_map.items():
            if k.co_name.startswith("#") or k.co_name.startswith("$"):
                states.append(v)
            elif k in CodeStatus().WITH_GRAPH_API:
                assert v.state == CodeState.WITH_GRAPH
            else:
                assert v.state == CodeState.WITHOUT_GRAPH
        # run_net, forward, loop body, resumed part2 in loop body
        assert len([v for v in states if v.state == CodeState.WITH_GRAPH]) == 4
        # resumed part1 in loop body
        assert (
            len([v for v in states if v.state == CodeState.WITHOUT_GRAPH]) == 1
        )

    def test_case_2(self):
        with strict_mode_guard(False):
            CodeStatus().clear()
            net = SimpleNet2()
            inp = paddle.rand((10, 10))
            self.assert_results(run_net, net, inp)
            code_map = CodeStatus().code_map
            states = []
            for k, v in code_map.items():
                if k.co_name.startswith("#") or k.co_name.startswith("$"):
                    states.append(v)
                elif k in CodeStatus().WITH_GRAPH_API:
                    assert v.state == CodeState.WITH_GRAPH
                else:
                    assert v.state == CodeState.WITHOUT_GRAPH
            # no graph found because fallback (paddle api will not enter simulate)
            assert (
                len([v for v in states if v.state == CodeState.WITH_GRAPH]) == 0
            )


def no_skip_func_0(x):
    return x + 1


def skipped_func_0():
    pass


def skipped_func_1(x):
    return x + 1


def skipped_func_2(x):
    return no_skip_func_0(x)


def call_skipped_func_0(x):
    for i in range(15):
        skipped_func_0()
        x = skipped_func_1(x)
        x = skipped_func_2(x)
    return x


skip_function(skipped_func_0)
skip_function(skipped_func_1)
skip_function(skipped_func_2)
skip_function(call_skipped_func_0)


class TestDisableSkippedFrame(TestCaseBase):
    def test_case_0(self):
        CodeStatus().clear()
        x = paddle.to_tensor([1])
        self.assert_results(call_skipped_func_0, x)
        code_map = CodeStatus().code_map
        assert (
            code_map[skipped_func_0.__code__].state == CodeState.WITHOUT_GRAPH
        )
        assert (
            code_map[skipped_func_1.__code__].state == CodeState.WITHOUT_GRAPH
        )
        assert code_map[skipped_func_2.__code__].state == CodeState.WITH_GRAPH


if __name__ == "__main__":
    unittest.main()
