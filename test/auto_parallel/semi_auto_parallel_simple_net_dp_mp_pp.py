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

import os

from semi_auto_parallel_simple_net import (
    DemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist


class TestSimpleNetHybridStrategyForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        self._pp_mesh0 = dist.ProcessMesh(
            [[0, 1], [2, 3]], dim_names=["x", "y"]
        )
        self._pp_mesh1 = dist.ProcessMesh(
            [[4, 5], [6, 7]], dim_names=["x", "y"]
        )
        self.pp_reshard_dist_attr = dist.DistAttr(
            mesh=self._pp_mesh1, sharding_specs=["x", "y"]
        )

        paddle.set_device(self._backend)

        self.init_input_data()
        self.init_single_card_net_result()

    def dp_mp_pp_shard_fn(self, layer_name, layer, process_mesh):
        if layer_name == 'linear_0':
            # shard_layer doens't support cross-mesh now.
            # input process_mesh of pp_shard_fn is useless,
            # it's defined just for unified format.
            weight_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh0, sharding_specs=[None, 'y']
            )
            bias_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh0, sharding_specs=[None]
            )
            layer.weight = dist.shard_tensor(
                layer.weight, dist_attr=weight_dist_attr
            )
            layer.bias = dist.shard_tensor(layer.bias, dist_attr=bias_dist_attr)
        elif layer_name == 'linear_1':
            weight_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh1, sharding_specs=['y', None]
            )
            bias_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh1, sharding_specs=[None]
            )
            layer.weight = dist.shard_tensor(
                layer.weight, dist_attr=weight_dist_attr
            )
            layer.bias = dist.shard_tensor(layer.bias, dist_attr=bias_dist_attr)

    def test_dp_mp_pp_demo_net(self):
        model = dist.shard_layer(
            DemoNet(
                "dp_mp_pp_hybrid_strategy",
                is_pp=True,
                pp_reshard_dist_attr=self.pp_reshard_dist_attr,
            ),
            self._pp_mesh0,
            self.dp_mp_pp_shard_fn,
        )

        (
            self.dp_mp_pp_loss,
            self.dp_mp_pp_parameters,
        ) = self.run_dynamic(model, shard_input=True, is_pp=True)

        rank = dist.get_rank()
        # TODO(GhostScreaming): DistTensor.numpy() doesn't support
        # cross-mesh now, ReshardXToReplicated function in eager_method
        # needs to be fixed later.
        if rank in [0, 1, 2, 3]:
            # linear_0 weight and bias
            self.check_tensor_eq(
                self.dp_mp_pp_parameters[0], self.base_parameters[0]
            )
            self.check_tensor_eq(
                self.dp_mp_pp_parameters[1], self.base_parameters[1]
            )
        else:
            self.check_tensor_eq(self.dp_mp_pp_loss, self.base_loss)
            # linear_1 weight and bias
            self.check_tensor_eq(
                self.dp_mp_pp_parameters[2], self.base_parameters[2]
            )
            self.check_tensor_eq(
                self.dp_mp_pp_parameters[3], self.base_parameters[3]
            )

    def run_test_case(self):
        self.test_dp_mp_pp_demo_net()


if __name__ == '__main__':
    TestSimpleNetHybridStrategyForSemiAutoParallel().run_test_case()
