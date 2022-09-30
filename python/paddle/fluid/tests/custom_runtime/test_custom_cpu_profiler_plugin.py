# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import site
import unittest
import numpy as np
import tempfile


class TestCustomCPUProfilerPlugin(unittest.TestCase):

    def setUp(self):
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        cmd = 'cd {} \
            && git clone {} \
            && cd PaddleCustomDevice \
            && git fetch origin \
            && git checkout {} -b dev \
            && cd backends/custom_cpu \
            && mkdir build && cd build && cmake .. && make -j8'.format(
            self.temp_dir.name, os.getenv('PLUGIN_URL'),
            os.getenv('PLUGIN_TAG'))
        os.system(cmd)

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
            cur_dir, '{}/PaddleCustomDevice/backends/custom_cpu/build'.format(
                self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()
        del os.environ['CUSTOM_DEVICE_ROOT']

    def test_custom_device(self):
        import paddle
        with paddle.fluid.framework._test_eager_guard():
            self._test_custom_profiler()

    def _test_custom_profiler(self):
        import paddle
        import paddle.profiler as profiler

        paddle.set_device('custom_cpu')

        x = paddle.to_tensor([1, 2, 3])
        p = profiler.Profiler(targets=[
            profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE
        ])
        p.start()
        for iter in range(10):
            x = x + 1
            p.step()
        p.stop()
        p.summary()


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
