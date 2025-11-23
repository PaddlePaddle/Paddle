# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from contextlib import contextmanager

import numpy as np
from dygraph_to_static_utils import Dy2StTestBase

import paddle
from paddle.jit.dy2static.utils import CUDAGraphState

SEED = 2025
np.random.seed(2025)
GLOBAL_GRAPH_WITH_BUFFER = None


class GraphWithBuffer:
    def __init__(self, inputs, outputs):
        self.inputs_buffer = inputs
        self.outputs_buffer = outputs

    def set_inputs_buffer(self, inputs):
        assert len(self.inputs_buffer) == len(inputs)
        for i, _ in enumerate(inputs):
            self.inputs_buffer[i][:] = inputs[i]

    def get_inputs(self):
        return self.inputs_buffer

    def get_real_outputs(self):
        return self.outputs_buffer

    def get_outputs(self):
        return [out.clone() for out in self.outputs_buffer]


def capture_run_impl(original_run_impl, inputs, parameters, attrs):
    prog_attrs, cuda_graph_attrs = attrs
    cuda_graph_attrs |= {
        "cuda_graph_state": CUDAGraphState.CAPTURE,
        "cuda_graph_dispatch_key": inputs[0].shape[0],
    }
    outputs = original_run_impl(
        inputs, parameters, (prog_attrs, cuda_graph_attrs)
    )

    global GLOBAL_GRAPH_WITH_BUFFER
    if GLOBAL_GRAPH_WITH_BUFFER is None:
        GLOBAL_GRAPH_WITH_BUFFER = GraphWithBuffer(inputs, outputs)

    return outputs


def replay_run_impl(original_run_impl, inputs, parameters, attrs):
    prog_attrs, cuda_graph_attrs = attrs
    cuda_graph_attrs |= {
        "cuda_graph_state": CUDAGraphState.REPLAY,
        "cuda_graph_dispatch_key": inputs[0].shape[0],
    }
    global GLOBAL_GRAPH_WITH_BUFFER
    assert GLOBAL_GRAPH_WITH_BUFFER is not None
    GLOBAL_GRAPH_WITH_BUFFER.set_inputs_buffer(inputs)

    _ = original_run_impl(
        GLOBAL_GRAPH_WITH_BUFFER.get_inputs(),
        parameters,
        (prog_attrs, cuda_graph_attrs),
    )

    return GLOBAL_GRAPH_WITH_BUFFER.get_outputs()


@contextmanager
def capture_run_impl_guard():
    with paddle.jit.dy2static.pir_partial_program.replace_run_impl_guard(
        capture_run_impl,
    ):
        yield


@contextmanager
def replay_run_impl_guard():
    with paddle.jit.dy2static.pir_partial_program.replace_run_impl_guard(
        replay_run_impl,
    ):
        yield


@unittest.skipIf(
    (not paddle.is_compiled_with_cuda()) or paddle.is_compiled_with_rocm(),
    "Skipped on non-GPU devices and ROCm devices(DCU) as this test requires NVIDIA CUDA Graph.",
)
class TestCUDAGraph(Dy2StTestBase):
    def initialize(self):
        global GLOBAL_GRAPH_WITH_BUFFER
        GLOBAL_GRAPH_WITH_BUFFER = None

        def func(x, y):
            return x + y

        self.fn = func
        self.static_fn = paddle.jit.to_static(func)

    def test_capture_replay(self):
        self.initialize()
        x = paddle.randn([2, 2, 3, 3], dtype='float32')
        y = paddle.randn([2, 2, 3, 3], dtype='float32')
        with capture_run_impl_guard():
            _ = self.static_fn(x, y)

        a = paddle.randn([2, 2, 3, 3], dtype='float32')
        b = paddle.randn([2, 2, 3, 3], dtype='float32')
        with replay_run_impl_guard():
            c = self.static_fn(a, b)

        np.testing.assert_allclose(self.fn(a, b), c)


if __name__ == "__main__":
    unittest.main()
