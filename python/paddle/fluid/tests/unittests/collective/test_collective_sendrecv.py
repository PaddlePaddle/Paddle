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

<<<<<<< HEAD
import unittest

from test_collective_base import TestDistBase

import paddle

=======
from __future__ import print_function
import unittest
import numpy as np
import paddle

from test_collective_base import TestDistBase

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
paddle.enable_static()


class TestSendRecvOp(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        pass

    def test_sendrecv(self):
        self.check_with_place("collective_sendrecv_op.py", "sendrecv")

    def test_sendrecv_dynamic_shape(self):
<<<<<<< HEAD
        self.check_with_place(
            "collective_sendrecv_op_dynamic_shape.py", "sendrecv_dynamic_shape"
        )

    def test_sendrecv_array(self):
        self.check_with_place(
            "collective_sendrecv_op_array.py", "sendrecv_array"
        )
=======
        self.check_with_place("collective_sendrecv_op_dynamic_shape.py",
                              "sendrecv_dynamic_shape")

    def test_sendrecv_array(self):
        self.check_with_place("collective_sendrecv_op_array.py",
                              "sendrecv_array")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
