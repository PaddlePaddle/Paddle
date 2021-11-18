# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid

paddle.enable_static()


class TestFleetExecutor(unittest.TestCase):
    def run_fleet_executor(self, place):
        exe = paddle.static.Executor(place)
        empty_program = paddle.static.Program()
        with fluid.program_guard(empty_program, empty_program):
            x = fluid.layers.data(name='x', shape=[1], dtype=paddle.float32)
        empty_program._pipeline_opt = {
            "fleet_opt": {},
            "section_program": empty_program
        }
        exe.run(empty_program, feed={'x': [1]})

    def test_executor_on_single_device(self):
        places = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.run_fleet_executor(place)


if __name__ == "__main__":
    unittest.main()
