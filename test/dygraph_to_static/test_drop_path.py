# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle


def drop_path(x, training=False):
    if not training:
        return x
    else:
        return 2 * x


class DropPath(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return drop_path(x, self.training)


class TestTrainEval(Dy2StTestBase):
    def test_train_and_eval(self):
        model = paddle.jit.to_static(DropPath())
        x = paddle.to_tensor([1, 2, 3]).astype("int64")
        eval_out = x.numpy()
        train_out = x.numpy() * 2
        model.train()
        np.testing.assert_allclose(model(x).numpy(), train_out, rtol=1e-05)
        model.eval()
        np.testing.assert_allclose(model(x).numpy(), eval_out, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
