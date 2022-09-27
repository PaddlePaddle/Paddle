#   copyright (c) 2020 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import unittest
import os
import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.static as static


class Test_XPU_Places(unittest.TestCase):

    def assert_places_equal(self, places0, places1):
        self.assertEqual(len(places0), len(places1))
        for place0, place1 in zip(places0, places1):
            self.assertEqual(type(place0), type(place1))
            self.assertEqual(place0.get_device_id(), place1.get_device_id())

    def test_check_preset_envs(self):
        if core.is_compiled_with_xpu():
            os.environ["FLAGS_selected_xpus"] = "0"
            place_list = static.xpu_places()
            self.assert_places_equal([fluid.XPUPlace(0)], place_list)

    def test_check_no_preset_envs(self):
        if core.is_compiled_with_xpu():
            place_list = static.xpu_places(0)
            self.assert_places_equal([fluid.XPUPlace(0)], place_list)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
