#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.layers.device import get_places
from decorator_helper import prog_scope
import unittest


class TestGetPlaces(unittest.TestCase):

    @prog_scope()
    def check_get_cpu_places(self):
        places = get_places()
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(fluid.default_main_program())
        self.assertEqual(places.type, fluid.core.VarDesc.VarType.PLACE_LIST)

    @prog_scope()
    def check_get_gpu_places(self):
        places = get_places(device_type='CUDA')
        gpu = fluid.CUDAPlace(0)
        exe = fluid.Executor(gpu)
        exe.run(fluid.default_main_program())
        self.assertEqual(places.type, fluid.core.VarDesc.VarType.PLACE_LIST)

    def test_main(self):
        if core.is_compiled_with_cuda():
            self.check_get_gpu_places()
        self.check_get_cpu_places()


if __name__ == '__main__':
    unittest.main()
