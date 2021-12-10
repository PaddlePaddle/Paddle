#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
import paddle
import paddle.fluid as fluid

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuPlace(unittest.TestCase):
    def test_ipu_place(self):
        num_devices = fluid.core.get_ipu_device_count()
        self.assertGreater(num_devices, 0)

        for i in range(num_devices):
            place = paddle.IPUPlace()
            p = fluid.core.Place()
            p.set_place(place)
            self.assertTrue(p.is_ipu_place())

    def test_ipu_set_device(self):
        num_devices = fluid.core.get_ipu_device_count()
        self.assertGreater(num_devices, 0)

        for i in range(num_devices):
            paddle.set_device('ipu')
            device = paddle.get_device()
            self.assertTrue(device == "ipus:{{0-{}}}".format(num_devices - 1))


if __name__ == '__main__':
    unittest.main()
