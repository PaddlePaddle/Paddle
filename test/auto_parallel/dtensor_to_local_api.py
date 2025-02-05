# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.distributed as dist
from paddle.distributed import Partial
from paddle.distributed.auto_parallel.api import dtensor_to_local


class TestDtensorToLocalAPI:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._shard = eval(os.getenv("shard"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_cases(self):
        self.test_case_forward_backward()

    def test_case_forward_backward(self):
        a = paddle.ones(self._shape)
        a.stop_gradient = False

        input_tensor = dist.shard_tensor(a, self._mesh, [Partial()])
        input_tensor.register_hook(
            self.check_grad_mesh(
                input_tensor.process_mesh, input_tensor.placements
            )
        )

        tensor1 = dtensor_to_local(input_tensor)
        assert not tensor1.is_dist()

        tensor2 = tensor1 + 2
        tensor3 = tensor2 * 3
        tensor3.register_hook(self.check_grad_mesh(None, None))
        tensor3.backward()

    def check_grad_mesh(self, org_mesh, org_placements):
        def _check_mesh(grad):
            assert grad.process_mesh == org_mesh
            assert grad.placements == org_placements

        return _check_mesh


if __name__ == '__main__':
    TestDtensorToLocalAPI().run_test_cases()
