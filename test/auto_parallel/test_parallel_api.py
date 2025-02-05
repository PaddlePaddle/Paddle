#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed.auto_parallel.intermediate.parallel_base import (
    ParallelModel,
)


class PP(ParallelModel):
    def __init__(self, model):
        super().__init__(model)
        self.pp_parallelizer = self.pp_init

    def pp_init(self, model):
        return paddle.nn.Linear(2, 2)


class TP(ParallelModel):
    def __init__(self, model):
        super().__init__(model)
        self.tp_parallelizer = self.tp_init

    def tp_init(self, model):
        return paddle.nn.Linear(3, 3)


class SD(ParallelModel):
    def __init__(self, model):
        super().__init__(model)
        self.sharding_parallelizer = self.sd_init

    def sd_init(self, model):
        return paddle.nn.Linear(4, 4)


class TestStrategy:
    def test_recursive(self):
        model = paddle.nn.Linear(1, 1)
        pp = PP(model)
        data = paddle.rand([1, 2])
        pp(data)
        assert pp.model.weight.shape == [2, 2]
        model = paddle.nn.Linear(1, 1)
        tp = TP(PP(model))
        data = paddle.rand([1, 3])
        tp(data)
        assert tp.model.weight.shape == [3, 3]
        model = paddle.nn.Linear(1, 1)
        sd = SD(TP(PP(model)))
        data = paddle.rand([1, 4])
        sd(data)
        assert sd.model.weight.shape == [4, 4]


if __name__ == '__main__':
    unittest.main()
