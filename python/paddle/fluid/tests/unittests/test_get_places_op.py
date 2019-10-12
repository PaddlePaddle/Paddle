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

from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.layers.device import get_places
from decorator_helper import prog_scope
import unittest


class TestGetPlaces(unittest.TestCase):
    @prog_scope()
    def test_get_places(self):
        places = get_places()
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(fluid.default_main_program())
        self.assertEqual(places.type, fluid.core.VarDesc.VarType.PLACE_LIST)


if __name__ == '__main__':
    unittest.main()
