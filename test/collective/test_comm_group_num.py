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

import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu import mp_ops


class CommGroupNumTest(unittest.TestCase):
    def test_comm_group_num(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 2,
            "sharding_degree": 2,
            "order": ["dp", "pp", "sharding", "sep", "mp"],
        }
        fleet.init(is_collective=True, strategy=strategy)

        place = paddle.framework._current_expected_place()
        input = np.random.uniform(
            low=-2.0, high=2.0, size=(1, 4096, 16000)
        ).astype('float32')
        input = paddle.to_tensor(input, place=place)
        input.stop_gradient = False

        label = np.random.randint(
            low=1, high=29956, size=(1, 4096, 1), dtype='int64'
        )
        label = paddle.to_tensor(label, place=place)
        label.stop_gradient = True

        model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
        )
        loss = mp_ops._c_softmax_with_cross_entropy(
            input,
            label,
            group=model_parallel_group,
            ignore_index=-100,
        )


if __name__ == '__main__':
    unittest.main()
