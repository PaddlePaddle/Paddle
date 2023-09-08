#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from test_dist_sparse_tensor_load_sgd import TestSparseLoadProgram

import paddle
from paddle import base
from paddle.distributed.fleet import fleet


class TestSparseLoadProgramAdagrad(TestSparseLoadProgram):
    """
    Test Sparse load operator.
    """

    def test_server_init(self):
        scope, train_program, startup_program, loss = self.net()
        with base.scope_guard(scope):
            with base.program_guard(train_program, startup_program):
                optimizer = paddle.optimizer.Adam(1e-3)
                optimizer = fleet.distributed_optimizer(
                    optimizer, self.strategy
                )
                optimizer.minimize(loss)
                fleet.init_server()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
