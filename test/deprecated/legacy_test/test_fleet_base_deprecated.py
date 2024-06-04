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
from paddle import base
from paddle.distributed import fleet


class TestFleetBaseSingleError(unittest.TestCase):
    def gen_data(self):
        return {
            "x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(2, size=(128, 1)).astype('int64'),
        }

    def test_single_run_collective_minimize(self):
        def test_single_error():
            input_x = paddle.static.data(
                name="x", shape=[-1, 32], dtype='float32'
            )
            input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')

            fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
            prediction = paddle.static.nn.fc(
                x=fc_1, size=2, activation='softmax'
            )
            cost = paddle.nn.functional.cross_entropy(
                input=prediction,
                label=input_y,
                reduction='none',
                use_softmax=False,
            )
            avg_cost = paddle.mean(x=cost)
            fleet.init(is_collective=True)

        # in non_distributed mode(use `python` to launch), raise error if has multi cards
        if (
            base.core.is_compiled_with_cuda()
            and base.core.get_cuda_device_count() > 1
        ):
            self.assertRaises(ValueError, test_single_error)
        else:
            test_single_error()


if __name__ == "__main__":
    unittest.main()
