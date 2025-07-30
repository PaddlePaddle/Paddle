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
from paddle.distributed.auto_parallel.api import dtensor_from_local


class TestDtensorFromLocalAPI:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._shard = eval(os.getenv("shard"))
        self._mesh = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"])

    def run_test_cases(self):
        self.test_case_forward_backward()

    def test_case_forward_backward(self):
        a = paddle.ones([10, 20])
        a.stop_gradient = False

        tensor1 = a + 3
        assert not tensor1.is_dist()
        tensor1.register_hook(self.check_grad_mesh(None, None))

        mesh = self._mesh
        tensor2 = dtensor_from_local(
            tensor1, mesh, [dist.Shard(0), dist.Replicate()]
        )

        assert tensor2.is_dist()
        assert tensor2.process_mesh == mesh
        assert tensor2.placements == [dist.Shard(0), dist.Replicate()]
        tensor2.register_hook(
            self.check_grad_mesh(mesh, [dist.Shard(0), dist.Replicate()])
        )

        tensor3 = tensor2 * 3
        tensor3.register_hook(
            self.check_grad_mesh(mesh, [dist.Shard(0), dist.Replicate()])
        )
        tensor4 = tensor3 + 4

        tensor4.backward()

    def check_grad_mesh(self, mesh, placements):
        def _check_mesh(grad):
            if mesh is None and placements is None:
                assert not grad.is_dist(), "grad.is_dist() is not False"
            else:
                assert (
                    grad.process_mesh == mesh
                ), "grad.process_mesh is not equal to mesh"
                assert (
                    grad.placements == placements
                ), "grad.placements is not equal to placements"

        return _check_mesh


if __name__ == '__main__':
    TestDtensorFromLocalAPI().run_test_cases()
