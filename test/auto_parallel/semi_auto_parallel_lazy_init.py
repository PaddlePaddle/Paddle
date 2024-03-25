# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import os

import paddle
import paddle.distributed as dist
from paddle import LazyGuard


class TestSemiAutoParallelLazyInit:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._placements_type = os.getenv("_placements_type")
        self._seed = eval(os.getenv("seed"))
        if self._placements_type == "DP":
            self._mesh_weight = dist.ProcessMesh([0, 1], dim_names=["x"])
            self._mesh_bias = dist.ProcessMesh([0, 1], dim_names=["x"])
            self._placements_weight = [dist.Replicate()]
            self._placements_bias = [dist.Replicate()]
        elif self._placements_type == "PP":
            self._mesh_weight = dist.ProcessMesh([0], dim_names=["x"])
            self._mesh_bias = dist.ProcessMesh([1], dim_names=["x"])
            self._placements_weight = [dist.Replicate()]
            self._placements_bias = [dist.Replicate()]
        elif self._placements_type == "MP":
            self._mesh_weight = dist.ProcessMesh([0, 1], dim_names=["x"])
            self._mesh_bias = dist.ProcessMesh([0, 1], dim_names=["x"])
            self._placements_weight = [dist.Shard(1)]
            self._placements_bias = [dist.Shard(0)]

    def test_different_xavier(self):
        paddle.distributed.auto_parallel.parallel_manual_seed(self._seed)
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal()
        )
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform()
        )
        with LazyGuard():
            linear = paddle.nn.Linear(
                10, 10, weight_attr=weight_attr, bias_attr=bias_attr
            )
            linear.weight = dist.shard_tensor(
                linear.weight, self._mesh_weight, self._placements_weight
            )
            linear.bias = dist.shard_tensor(
                linear.bias, self._mesh_bias, self._placements_bias
            )
        for param in linear.parameters():
            param.initialize()
            logging.info(param)

    def test_constant(self):
        paddle.distributed.auto_parallel.parallel_manual_seed(self._seed)
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Constant(2.0)
        )
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Constant(1.0)
        )
        with LazyGuard():
            linear = paddle.nn.Linear(
                10, 10, weight_attr=weight_attr, bias_attr=bias_attr
            )
            linear.weight = dist.shard_tensor(
                linear.weight, self._mesh_weight, self._placements_weight
            )
            linear.bias = dist.shard_tensor(
                linear.bias, self._mesh_bias, self._placements_bias
            )
        for param in linear.parameters():
            param.initialize()
            logging.info(param)

    def test_placements(self):
        paddle.distributed.auto_parallel.parallel_manual_seed(self._seed)
        with LazyGuard():
            linear = paddle.nn.Linear(10, 10)
            linear.weight = dist.shard_tensor(
                linear.weight, self._mesh_weight, self._placements_weight
            )
            linear.bias = dist.shard_tensor(
                linear.bias, self._mesh_bias, self._placements_bias
            )
        for param in linear.parameters():
            assert not param._is_initialized()
            param.initialize()
            logging.info(param)

        if self._placements_type == "DP":
            assert linear.weight._is_initialized()
            assert linear.bias._is_initialized()
            local_weight_md5 = linear.weight._local_value()._md5sum()
            mesh0 = dist.ProcessMesh([0], dim_names=["x"])
            mesh1 = dist.ProcessMesh([1], dim_names=["x"])
            tmp = paddle.distributed.auto_parallel.api.dtensor_from_local(
                linear.weight._local_value(),
                mesh0 if dist.get_rank() == 0 else mesh1,
                [dist.Replicate()],
            )
            tmp = dist.reshard(
                tmp,
                mesh1 if dist.get_rank() == 0 else mesh0,
                [dist.Replicate()],
            )
            tmp_md5 = tmp._local_value()._md5sum()
            assert local_weight_md5 == tmp_md5
        elif self._placements_type == "PP":
            if dist.get_rank() == 0:
                assert linear.weight._is_initialized()
                assert not linear.bias._is_initialized()
            else:
                assert not linear.weight._is_initialized()
                assert linear.bias._is_initialized()
        elif self._placements_type == "MP":
            assert linear.weight._is_initialized()
            assert linear.bias._is_initialized()
            assert linear.weight._local_shape == [10, 5]
            assert linear.bias._local_shape == [5]

    def test_unbalance_mp(self):
        paddle.distributed.auto_parallel.parallel_manual_seed(self._seed)
        with LazyGuard():
            linear = paddle.nn.Linear(11, 11)
            linear.weight = dist.shard_tensor(
                linear.weight, self._mesh_weight, self._placements_weight
            )
            linear.bias = dist.shard_tensor(
                linear.bias, self._mesh_bias, self._placements_bias
            )
        for param in linear.parameters():
            assert not param._is_initialized()
            param.initialize()
            assert param._is_initialized()

        if dist.get_rank() == 0:
            assert linear.weight._local_shape == [11, 6]
            assert linear.bias._local_shape == [6]
        else:
            assert linear.weight._local_shape == [11, 5]
            assert linear.bias._local_shape == [5]

    def run_test_case(self):
        self.test_placements()
        self.test_different_xavier()
        self.test_constant()
        if self._placements_type == "MP":
            self.test_unbalance_mp()


if __name__ == '__main__':
    TestSemiAutoParallelLazyInit().run_test_case()
