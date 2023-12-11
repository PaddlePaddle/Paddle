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

from auto_parallel.semi_auto_parallel_simple_net import (
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
        self._ckpt_path = os.getenv("ckpt_path")
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

        paddle.set_device(self._backend)

        self.set_random_seed(self._seed)
        self.init_single_card_net_result()

    def test_dp_mp_demo_net(self):
        self.set_random_seed(self._seed)
        model = dist.shard_layer(
            DemoNet("dp_mp_hybrid_strategy"), self._mesh, self.shard_fn
        )

        (
            self.dp_mp_loss,
            self.dp_mp_parameters,
        ) = self.run_dynamic(model, shard_input=True)

        self.check_tensor_eq(self.dp_mp_loss, self.base_loss)
        for param, param_base in zip(
            self.dp_mp_parameters, self.base_parameters
        ):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

        # save load
        state_dict = model.state_dict()
        local_state_dict = {}
        for k, v in state_dict.items():
            local_state_dict[k] = v._local_value().clone()
        paddle.distributed.save_state_dict(state_dict, self._ckpt_path)
        for k, v in state_dict.items():
            v._local_value().add_(paddle.ones_like(v._local_value()))
        paddle.distributed.load_state_dict(state_dict, self._ckpt_path)
        for k, v in state_dict.items():
            assert k in local_state_dict, k
            self.check_tensor_eq(v._local_value(), local_state_dict[k])

    def run_test_case(self):
        self.test_dp_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetHybridStrategyForSemiAutoParallel().run_test_case()
