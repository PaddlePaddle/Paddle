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

from auto_parallel.hybrid_strategy.semi_auto_save_state_dict import (
    check_structure_name_mapping,
)
from auto_parallel.semi_auto_parallel_simple_net import (
    DemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle.distributed import Replicate, Shard


class TestSimpleNetHybridStrategyForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._ckpt_path = os.getenv("ckpt_path")
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        self._pp_mesh0 = dist.ProcessMesh(
            [[0, 1], [2, 3]], dim_names=["x", "y"]
        )
        self._pp_mesh1 = dist.ProcessMesh(
            [[4, 5], [6, 7]], dim_names=["x", "y"]
        )
        self.pp_reshard_dist_attr = (self._pp_mesh1, [Shard(0), Shard(1)])

        paddle.set_device(self._backend)

        self.set_random_seed(self._seed)
        self.init_single_card_net_result()

    def dp_mp_pp_shard_fn(self, layer_name, layer, process_mesh):
        if layer_name == 'linear_0':
            # shard_layer doesn't support cross-mesh now.
            # input process_mesh of pp_shard_fn is useless,
            # it's defined just for unified format.
            layer.weight = dist.shard_tensor(
                layer.weight, self._pp_mesh0, [Replicate(), Shard(1)]
            )
            if layer.bias is not None:
                layer.bias = dist.shard_tensor(
                    layer.bias, self._pp_mesh0, [Replicate(), Replicate()]
                )
        elif layer_name == 'linear_1':
            layer.weight = dist.shard_tensor(
                layer.weight, self._pp_mesh1, [Replicate(), Shard(0)]
            )
            if layer.bias is not None:
                layer.bias = dist.shard_tensor(
                    layer.bias, self._pp_mesh1, [Replicate(), Replicate()]
                )
        elif layer_name == 'norm':
            layer.weight = dist.shard_tensor(
                layer.weight, self._pp_mesh1, [Replicate(), Replicate()]
            )
            if layer.bias is not None:
                layer.bias = dist.shard_tensor(
                    layer.bias, self._pp_mesh1, [Replicate(), Replicate()]
                )

    def test_dp_mp_pp_demo_net(self):
        self.set_random_seed(self._seed)
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
                self.dp_mp_pp_parameters[0], self.base_parameters[0], rtol=2e-4
            )

        else:
            self.check_tensor_eq(self.dp_mp_pp_loss, self.base_loss, rtol=1e-4)
            self.check_tensor_eq(
                self.dp_mp_pp_parameters[1], self.base_parameters[1], rtol=1e-4
            )
            self.check_tensor_eq(
                self.dp_mp_pp_parameters[2], self.base_parameters[2], rtol=2e-5
            )
            self.check_tensor_eq(
                self.dp_mp_pp_parameters[3], self.base_parameters[3], rtol=2e-4
            )

        # save load
        state_dict = model.state_dict()
        paddle.distributed.save_state_dict(state_dict, self._ckpt_path)
        paddle.distributed.barrier()
        check_structure_name_mapping(self._ckpt_path, state_dict)
        expected_local_state_dict = {}
        need_load_state_dict = {}
        for k, v in state_dict.items():
            expected_local_state_dict[k] = (
                v._local_value().clone() if v._is_initialized() else None
            )
            need_load_state_dict[k] = (
                paddle.zeros_like(v) if v._is_initialized() else v
            )
        model.set_state_dict(need_load_state_dict)
        paddle.distributed.load_state_dict(
            need_load_state_dict, self._ckpt_path
        )
        model.set_state_dict(need_load_state_dict)
        state_dict = model.state_dict()
        for k, v in state_dict.items():
            assert k in expected_local_state_dict, k
            if v._is_initialized():
                self.check_tensor_eq(
                    v._local_value(), expected_local_state_dict[k]
                )

    def run_test_case(self):
        self.test_dp_mp_pp_demo_net()


if __name__ == '__main__':
    TestSimpleNetHybridStrategyForSemiAutoParallel().run_test_case()
