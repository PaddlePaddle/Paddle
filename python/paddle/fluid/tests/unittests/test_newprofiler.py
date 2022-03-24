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

        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(
            x_value, stop_gradient=False, place=paddle.CPUPlace())
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], ) as prof:
            y = x / 2.0
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=(1, 2)) as prof:
            with profiler.RecordEvent(name='test'):
                y = x / 2.0
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=profiler.make_scheduler(
                    closed=0, ready=1, record=1, repeat=1),
                on_trace_ready=my_trace_back) as prof:
            y = x / 2.0
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=profiler.make_scheduler(
                    closed=0, ready=0, record=2, repeat=1),
                on_trace_ready=my_trace_back) as prof:
            for i in range(3):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=lambda x: profiler.ProfilerState.RECORD_AND_RETURN,
                on_trace_ready=my_trace_back) as prof:
            for i in range(2):
                y = x / 2.0
                prof.step()

        def my_sheduler(num_step):
            if num_step % 5 < 2:
                return profiler.ProfilerState.RECORD_AND_RETURN
            elif num_step % 5 < 3:
                return profiler.ProfilerState.READY
            elif num_step % 5 < 4:
                return profiler.ProfilerState.RECORD
            else:
                return profiler.ProfilerState.CLOSED

        def my_sheduler1(num_step):
            if num_step % 5 < 2:
                return profiler.ProfilerState.RECORD
            elif num_step % 5 < 3:
                return profiler.ProfilerState.READY
            elif num_step % 5 < 4:
                return profiler.ProfilerState.RECORD
            else:
                return profiler.ProfilerState.CLOSED

        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=lambda x: profiler.ProfilerState.RECORD_AND_RETURN,
                on_trace_ready=my_trace_back) as prof:
            for i in range(2):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=my_sheduler,
                on_trace_ready=my_trace_back) as prof:
            for i in range(5):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=my_sheduler1) as prof:
            for i in range(5):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=profiler.make_scheduler(
                    closed=1, ready=1, record=2, repeat=1, skip_first=1),
                on_trace_ready=my_trace_back) as prof:
            for i in range(5):
                y = x / 2.0
                paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
                prof.step()

        prof.export(path='./test_profiler_pb.pb', format='pb')
        prof.summary()
        result = profiler.utils.load_profiler_result('./test_profiler_pb.pb')


if __name__ == '__main__':
    unittest.main()
