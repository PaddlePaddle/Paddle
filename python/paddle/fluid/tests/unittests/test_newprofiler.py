#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import os
import numpy as np
import gc

import paddle
import paddle.profiler as profiler


class TestProfiler(unittest.TestCase):
    def test_profiler(self):
        repeat = 20
        #test_profiler_cpu
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU], scheduler=[10, 20])
        prof.start()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False)
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        for i in range(repeat):
            y = x / 2.0
            paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
            prof.step()
        prof.stop()
        #test_profiler_gpu
        prof = None
        gc.enable()
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.GPU], scheduler=[10, 20])
        prof.start()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False)
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        for i in range(repeat):
            y = x / 2.0
            paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
            prof.step()
        prof.stop()
        prof.summary()

        #test_profiler_both
        prof = None
        gc.enable()
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.GPU, profiler.ProfilerTarget.CPU],
            scheduler=[10, 20])
        prof.start()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False)
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        for i in range(repeat):
            y = x / 2.0
            paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
            prof.step()
        prof.stop()
        prof.summary()

        # test_profiler_sheduler
        prof = None
        gc.enable()
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.GPU, profiler.ProfilerTarget.CPU],
            scheduler=profiler.make_scheduler(
                closed=1, ready=1, record=3, repeat=1))
        prof.start()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False)
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        for i in range(repeat):
            y = x / 2.0
            paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
            prof.step()
        prof.stop()
        prof.summary()

        # test_profiler_logger
        prof = None
        gc.enable()
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.GPU, profiler.ProfilerTarget.CPU],
            scheduler=profiler.make_scheduler(
                closed=1, ready=1, record=3, repeat=1),
            on_trace_ready=profiler.export_chrome_tracing('./test_profiler'))
        prof.start()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False)
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        for i in range(repeat):
            y = x / 2.0
            paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
            prof.step()
        prof.stop()
        prof.summary()
        prof = None
        gc.enable()
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.GPU, profiler.ProfilerTarget.CPU],
            scheduler=profiler.make_scheduler(
                closed=1, ready=1, record=3, repeat=1), )
        prof.start()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False)
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        for i in range(repeat):
            y = x / 2.0
            paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
            prof.step()
        prof.stop()
        prof.export(path='./test_profiler_pb.pb', format='pb')
        prof.summary()
        result = profiler.utils.LoadProfilerResult('./test_profiler_pb.pb')


if __name__ == '__main__':
    unittest.main()
