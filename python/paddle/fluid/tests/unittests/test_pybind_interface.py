#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core
from paddle import compat as cpt


class TestPybindInference(unittest.TestCase):

    # call get_op_attrs_default_value for c++ coverage rate
    def test_get_op_attrs_default_value(self):
        core.get_op_attrs_default_value(cpt.to_bytes("fill_constant"))

        # the default values of Op 'fill_constant'
        #
        # {"str_value": "",
        #  "force_cpu": false,
        #  "value": 1.0,
        #  "op_role_var": [],
        #  "shape": [],
        #  "op_namescope": "",
        #  "test_attr_1": 1.0,
        #  "op_callstack": [],
        #  "op_role": 4096}


if __name__ == '__main__':
    unittest.main()
