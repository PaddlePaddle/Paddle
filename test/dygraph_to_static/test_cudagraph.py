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

import paddle
from paddle.jit.dy2static.utils import CUDAGraphState

SEED = 2025
np.random.seed(2025)


class Dy2StCudaGraphManager:
    def __init__(self):
        self.state = CUDAGraphState.DISABLE
        self.captured_batch_size = set()
        self.batch_size = -1

    def run_impl(self, original_run_impl, inputs, parameters, attrs):
        prog_attrs, cuda_graph_attrs = attrs
        if self.state == CUDAGraphState.REPLAY:
            if self.batch_size not in self.captured_batch_size:
                self.state = CUDAGraphState.DISABLE
        elif self.state == CUDAGraphState.CAPTURE:
            self.captured_batch_size.add(self.batch_size)

        cuda_graph_attrs |= {
            "cuda_graph_state": self.state,
            "cuda_graph_dispatch_key": self.batch_size
            if self.state != CUDAGraphState.DISABLE
            else 0,
        }
        return original_run_impl(
            inputs, parameters, (prog_attrs, cuda_graph_attrs)
        )

    @contextmanager
    def run_impl_guard(self):
        with paddle.jit.dy2static.pir_partial_program.replace_run_impl_guard(
            self.run_impl,
        ):
            yield


class CudaGraphRunner:
    def __init__(self, runnable):
        self.runnable = runnable
        self.captured = False
        self.cuda_graph_manager = Dy2StCudaGraphManager()

    def run_static_model(self, x):
        if not self.captured:
            # Capture
            self.cuda_graph_manager.state = CUDAGraphState.CAPTURE
            self.cuda_graph_manager.batch_size = x.shape[0]
            self.captured = True
            with self.cuda_graph_manager.run_impl_guard():
                return self.runnable(x)

        # Replay
        assert self.captured
        self.cuda_graph_manager.state = CUDAGraphState.REPLAY
        self.cuda_graph_manager.batch_size = x.shape[0]
        with self.cuda_graph_manager.run_impl_guard():
            return self.runnable(x)


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda(), reason="Require CUDA."
)
class TestCUDAGraph1(unittest.TestCase):
    def initialize(self):
        self.fn = lambda x: x + x
        self.static_fn = paddle.jit.to_static(self.fn)
        self.x = paddle.rand([4, 3])

    def test_cuda_graph(self):
        self.initialize()
        runner = CudaGraphRunner(self.static_fn)
        # Captured
        runner.run_static_model(self.x)
        # Replay
        y_cg = runner.run_static_model(self.x)
        y_dy = self.fn(self.x)

        np.testing.assert_allclose(y_dy, y_cg)


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda(), reason="Require CUDA."
)
class TestCUDAGraph2(TestCUDAGraph1):
    def initialize(self):
        layer = paddle.nn.Conv2D(3, 3, 3)
        self.fn = layer
        self.static_fn = paddle.jit.to_static(self.fn)
        self.x = paddle.rand([2, 3, 32, 32])


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda(), reason="Require CUDA."
)
class TestCUDAGraph3(TestCUDAGraph1):
    def initialize(self):
        layer = paddle.nn.Linear(8, 4)
        self.fn = layer
        self.static_fn = paddle.jit.to_static(self.fn)
        self.x = paddle.rand([4, 8])


if __name__ == "__main__":
    unittest.main()
