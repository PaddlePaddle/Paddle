# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import (
    ColumnParallelLinear,
)
from paddle.nn import Layer


class SimpleMLP(Layer):
    def __init__(self, in_features=1024, out_features=1024):
        super().__init__()
        self.linear = ColumnParallelLinear(
            in_features, out_features, has_bias=False
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class TestLoadStateDictCastLogic:
    def __init__(self):
        self.aoa_config = {"aoa_statements": [os.getenv("aoa_statements")]}
        self.ckpt_path = tempfile.TemporaryDirectory().name
        self.in_features = 1024
        self.out_features = 2048

    def run_test(self):
        self.run_save_state_dict()
        model = SimpleMLP()
        model_cast = SimpleMLP()
        model_cast = paddle.amp.decorate(
            models=model_cast,
            optimizers=None,
            level="O2",
            dtype="float16",
        )
        sharded_state_dict = model.sharded_state_dict()
        sharded_state_dict_trans = model_cast.sharded_state_dict()
        dist.load_state_dict(sharded_state_dict, self.ckpt_path)
        dist.load_state_dict(
            sharded_state_dict_trans, self.ckpt_path, aoa_config=self.aoa_config
        )
        state_dict_1_after_load = model.state_dict()
        state_dict_2_after_load = model_cast.state_dict()

        np.testing.assert_array_equal(
            state_dict_1_after_load['linear.weight'].astype("float16"),
            state_dict_2_after_load['linear.weight'],
        )

    def setup_dist_env(self):
        fleet.init(is_collective=True)

    def run_save_state_dict(self):
        self.setup_dist_env()
        model = SimpleMLP()
        sharded_state_dict = model.sharded_state_dict()
        dist.save_state_dict(sharded_state_dict, self.ckpt_path)


if __name__ == '__main__':
    TestLoadStateDictCastLogic().run_test()
