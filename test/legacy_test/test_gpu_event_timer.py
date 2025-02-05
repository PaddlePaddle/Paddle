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

import unittest

import numpy as np

import paddle
from paddle.distributed.fleet.utils.timer_helper import get_timers, set_timers


class TestGPUEventTimer(unittest.TestCase):
    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return

        if paddle.is_compiled_with_rocm():
            return

        set_timers()
        key = "matmul"
        x = paddle.randn([1024, 1024])
        timers = get_timers()
        use_event = True

        timers(key, use_event=use_event).pre_alloc(5)

        for _ in range(2):
            for _ in range(3):
                timers(key, use_event=use_event).start()
                paddle.matmul(x, x)
                timers(key, use_event=use_event).stop()
            times = timers(key, use_event=use_event).elapsed_list(reset=False)
            assert isinstance(times, np.ndarray), times
            times2 = timers(key, use_event=use_event).elapsed_list(reset=False)
            np.testing.assert_array_equal(times, times2)
            timers.log(timers.timers.keys())

            assert timers(key, use_event=use_event).size() == 0
            assert timers(key, use_event=use_event).capacity() > 0
            timers(key, use_event=use_event).shrink_to_fit()
            assert timers(key, use_event=use_event).size() == 0
            assert timers(key, use_event=use_event).capacity() == 0


if __name__ == "__main__":
    unittest.main()
