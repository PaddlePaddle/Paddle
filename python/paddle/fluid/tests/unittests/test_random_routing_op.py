#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import op_test
import numpy as np
import unittest
import paddle
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.backward import append_backward
from paddle.distributed.models.moe import utils


def random_routing(topk_idx, topk_value, prob, topk=2):
    if topk == 2:
        new_topk_idx = np.copy(topk_idx)
        for i in range(len(topk_idx)):
            val = topk_value[i][1]
            if val * 2 < prob[i]:
                new_topk_idx[i][1] = -1
        return new_topk_idx
    else:
        raise RuntimeError("only topk=2 is supported now")


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNumberCountAPIFp32(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.init()

    def init(self):
        self.upper_range = 8
        self.x = np.random.randint(
            -1, self.upper_range, size=(200, 2)).astype('int64')
        self.prob = np.random.random((self.x.shape[0], )).astype(self.dtype)
        self.topk_value = np.random.random(self.x.shape).astype(self.dtype)
        self.out = random_routing(self.x, self.topk_value,
                                  self.prob).astype(self.dtype)
        self.place = paddle.CUDAPlace(0)

    def test_api_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        value = paddle.to_tensor(self.topk_value)
        prob = paddle.to_tensor(self.prob)
        out = utils._random_routing(x, value, prob)
        assert np.allclose(out.numpy(), self.out)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNumberCountAPIFp16(TestNumberCountAPIFp32):
    def setUp(self):
        self.dtype = "float16"
        self.init()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
