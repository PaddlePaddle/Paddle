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
import numpy as np

import paddle
import paddle.profiler as profiler


class TestProfiler(unittest.TestCase):
    def test_profiler(self):
        def my_trace_back(prof):
            profiler.export_chrome_tracing('./test_profiler_chrometracing/')(
                prof)
            profiler.export_protobuf('./test_profiler_pb/')(prof)

        repeat = 4
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(
            x_value, stop_gradient=False, place=paddle.CPUPlace())
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=profiler.make_scheduler(
                    closed=1, ready=1, record=1, repeat=1, skip_first=1),
                on_trace_ready=my_trace_back) as prof:
            for i in range(repeat):
                y = x / 2.0
                paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
                prof.step()
        prof.export(path='./test_profiler_pb.pb', format='pb')
        prof.summary()
        result = profiler.utils.LoadProfilerResult('./test_profiler_pb.pb')


if __name__ == '__main__':
    unittest.main()
