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

import time
import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot import psdb, symbolic_translate
from paddle.jit.sot.utils import StepInfoManager, StepState, cost_model_guard


def dyn_fast(x, net, iter_):
    for i in iter_:
        x = net(x)
    return x


def sot_fast_with_single_graph(x, net):
    if not psdb.in_sot():
        time.sleep(0.1)
    return x + 1


def sot_fast_with_multi_graph(x, net):
    if not psdb.in_sot():
        time.sleep(0.1)
    x = x + 1
    psdb.breakgraph()
    x = x + 2
    return x


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 10)

    def forward(self, x):
        if not psdb.in_sot():
            time.sleep(0.1)
        x = x / 3
        x = x + 5
        x = self.linear(x)
        return x


class TestCostModel(TestCaseBase):
    @cost_model_guard(True)
    def test_dyn_fast(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        sot_fn = symbolic_translate(dyn_fast)
        for i in range(60):
            sot_fn(x, net, iter(range(10)))

        state = StepInfoManager().step_record[dyn_fast.__code__].state
        assert state == StepState.RUN_DYN

    @cost_model_guard(True)
    def test_sot_fast_with_multi_graph(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        sot_fn = symbolic_translate(sot_fast_with_multi_graph)
        for i in range(30):
            sot_fn(x, net)

        state = (
            StepInfoManager()
            .step_record[sot_fast_with_multi_graph.__code__]
            .state
        )
        assert state == StepState.RUN_SOT

    @cost_model_guard(True)
    def test_sot_fast_with_single_graph(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        for i in range(30):
            symbolic_translate(sot_fast_with_single_graph)(x, net)

        state = (
            StepInfoManager()
            .step_record[sot_fast_with_single_graph.__code__]
            .state
        )
        assert state == StepState.RUN_SOT

    @cost_model_guard(True)
    def test_net(self):
        x = paddle.rand([10])
        net = Net()
        net = paddle.jit.to_static(net, full_graph=False)
        for i in range(30):
            x = net(x)

        state = StepInfoManager().step_record[Net.forward.__code__].state
        assert state == StepState.RUN_SOT


if __name__ == "__main__":
    unittest.main()
