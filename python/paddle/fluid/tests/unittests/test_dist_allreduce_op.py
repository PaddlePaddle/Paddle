# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from test_dist_base import TestDistBase

=======
from __future__ import print_function
import unittest
from test_dist_base import TestDistBase
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle

paddle.enable_static()


class TestDistMnistNCCL2(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._nccl2_reduce_layer = True

    def test_dist_train(self):
        import paddle.fluid as fluid
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("dist_allreduce_op.py", delta=1e-5)


if __name__ == '__main__':
    unittest.main()
