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

import paddle


def drop_path(x, training=False):
    if not training:
        return x
    else:
        return 2 * x


class DropPath(paddle.nn.Layer):

    def __init__(self):
        super(DropPath, self).__init__()

    @paddle.jit.to_static
    def forward(self, x):
        return drop_path(x, self.training)


class TestTrainEval(unittest.TestCase):

    def setUp(self):
        self.model = DropPath()

    def tearDown(self):
        pass

    def test_train_and_eval(self):
        x = paddle.to_tensor([1, 2, 3]).astype("int64")
        eval_out = x.numpy()
        train_out = x.numpy() * 2
        self.model.train()
        np.testing.assert_allclose(self.model(x).numpy(), train_out, rtol=1e-05)
        self.model.eval()
        np.testing.assert_allclose(self.model(x).numpy(), eval_out, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
