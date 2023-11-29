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
import re

import numpy as np

import paddle
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.nn import Linear, ReLU

logging.basicConfig(level="INFO", format="%(message)s")


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=1000):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)
        self._relu = ReLU()

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        y = self._relu(y)
        return y


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=200, linear_size=1000):
        self.num_samples = num_samples
        self.linear_size = linear_size
        self.samples = []
        for i in range(num_samples):
            img = np.random.rand(self.linear_size).astype('float32')
            self.samples.append(img)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return self.num_samples


def create_optimizer(model, use_pure_bf16, use_main_grad):
    if use_main_grad:
        assert use_pure_bf16
        model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.00001,
        weight_decay=0.00001,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        multi_precision=use_pure_bf16,
    )
    if use_main_grad:
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    return optimizer


def _extract_linear_order(param_names):
    # for param_names from model.state_dict, they are as like: ["_linear1.weight", "_linear1.bias"]
    # for master weight names from optimizer.state_dict, they are as like: ["linear_6.w_0", "linear_6.b_0"]
    param_order = []
    for name in param_names:
        param_id = re.findall(r"\d+", name)
        assert len(param_id) >= 1
        param_order.append(param_id[0])
    return list(set(param_order))


def compare_state_dict(state_dict_o1, state_dict_o2):
    master_weights = None
    if state_dict_o2.get("master_weights", None) is not None:
        master_weights = state_dict_o2["master_weights"]
    assert master_weights is not None
    master_weights_names = list(master_weights.keys())
    master_weight_order = _extract_linear_order(master_weights_names)
    master_weight_order.sort()

    param_names = list(state_dict_o1.keys())
    param_order = _extract_linear_order(param_names)
    param_order.sort()
    assert len(master_weight_order) <= len(param_order)

    param_order_dict = {}
    for i in range(len(master_weight_order)):
        param_order_dict[master_weight_order[i]] = param_order[i]

    param_master_pair = []

    # We assume the order of params in param_names and master_weights_names is the same.
    param_id = 0
    for master_weight_name in master_weights_names:
        master_weight_id = re.findall(r"\d+", master_weight_name)[0]
        param_id = param_order_dict[master_weight_id]
        for param_name in param_names:
            if (
                master_weight_name.endswith("w_0")
                and param_name.endswith("weight")
            ) or (
                master_weight_name.endswith("b_0")
                and param_name.endswith("bias")
            ):
                name_prefix = "linear" + param_id
                if name_prefix in param_name:
                    param_master_pair.append([param_name, master_weight_name])

    logging.info(f"-- param_names: {param_names}")
    logging.info(f"-- master_weights_names: {master_weights_names}")
    for pair in param_master_pair:
        param_name = pair[0]
        master_weight_name = pair[1]
        logging.info(f"-- compare {param_name} with {master_weight_name}")
        param_o1 = state_dict_o1[param_name]
        master_param_o2 = master_weights[master_weight_name]
        np.testing.assert_array_equal(param_o1.numpy(), master_param_o2.numpy())
