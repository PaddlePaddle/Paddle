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
"""
TestCases for Dataset,
including create, config, run, etc.
"""

import unittest

import numpy as np

import paddle


class TestVarInfo(unittest.TestCase):
    """TestCases for Dataset."""

    def test_var_info(self):
        """Testcase for get and set info for variable."""
        value = np.random.randn(1)
        var = paddle.static.create_global_var([1], value, "float32")
        var._set_info("name", "test")
        ret = var._get_info("name")
        assert ret == "test"
        ret = var._get_info("not_exist")
        assert ret is None


if __name__ == '__main__':
    unittest.main()
