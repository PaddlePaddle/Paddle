# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig, ServerRuntimeConfig, DistributeTranspiler
from paddle.fluid.transpiler.geo_sgd_transpiler import GeoSgdTranspiler


class TestTranspiler(unittest.TestCase):
    def test_init(self):
        config = DistributeTranspilerConfig()
        server_config = ServerRuntimeConfig()

        # test geo_transpiler exception
        self.assertRaises(Exception, DistributeTranspiler, dict())
        self.assertRaises(Exception, DistributeTranspiler, config, dict())

        transpiler = DistributeTranspiler(config, server_config)
        transpiler = DistributeTranspiler(config)
        transpiler = DistributeTranspiler(None, server_config)


class TestGeoSgdTranspiler(unittest.TestCase):
    def test_init(self):
        config = DistributeTranspilerConfig()
        server_config = ServerRuntimeConfig()

        # test geo_transpiler exception
        self.assertRaises(Exception, GeoSgdTranspiler, dict())
        self.assertRaises(Exception, GeoSgdTranspiler, config, dict())

        transpiler = GeoSgdTranspiler(config, server_config)
        transpiler = GeoSgdTranspiler(config)
        transpiler = GeoSgdTranspiler(None, server_config)


if __name__ == "__main__":
    unittest.main()
