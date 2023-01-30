#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import unittest

from test_dist_base import TestDistBase

=======
from __future__ import print_function
import unittest
from test_dist_base import TestDistBase

import os
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle

paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]


class TestStaticModelParallel(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl_comm_num = 1
        self._pipeline_mode = True

    def test_dist_static_model_parallel_fused_feedforward(self):
        import paddle.fluid as fluid
<<<<<<< HEAD

        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "static_model_parallel_fused_attention.py",
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name,
            )
=======
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("static_model_parallel_fused_attention.py",
                                  delta=1e-5,
                                  check_error_log=True,
                                  log_name=flag_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
