#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from scipy.special import logit

import paddle
from paddle import base


class TestSigmoidCrossEntropyWithLogitsOpGradWithAutoGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        paddle.seed(2023)
        self.places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))
        self.batch_size = 64
        self.num_classes = 20

        self.x = logit(
            np.random.uniform(0, 1, (self.batch_size, self.num_classes)).astype(
                "float32"
            )
        )

        self.lable = np.random.uniform(
            0, 1, (self.batch_size, self.num_classes)
        ).astype("float32")

        self.pos_weight = np.random.uniform(
            0, 1, (self.batch_size, self.num_classes)
        ).astype("float32")

    def test_check_grad_with_auto_grad(self):
        def fn_ref(x, label, weight):
            out = paddle._C_ops.sigmoid_cross_entropy_with_logits(
                x, label, weight, False, -100
            )
            loss = out.sum()
            loss.backward()
            return out, x.grad

        def fn_comp(x, label, weight):
            zeros = paddle.full((self.batch_size, self.num_classes), 0.0)
            t1 = paddle.where(x > 0, x, zeros)
            t2 = x * label
            t3 = paddle.log(1 + paddle.exp(-paddle.abs(x)))
            t4 = t1 - t2 + t3 * weight
            t5 = paddle.full((self.batch_size, self.num_classes), -100.0)
            out = paddle.where(label == t5, zeros, t4)
            loss = out.sum()
            loss.backward()
            return out, x.grad

        def cal(fn, place):
            x1 = paddle.to_tensor(self.x, stop_gradient=False, place=place)
            label1 = paddle.to_tensor(self.lable)
            pos_weight1 = paddle.to_tensor(self.pos_weight, place=place)
            res = fn(x1, label1, pos_weight1)
            return res

        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')

            ref = cal(fn_ref, p)
            actual = cal(fn_comp, p)

            for idx in range(len(ref)):
                np.testing.assert_allclose(
                    ref[idx].numpy(), actual[idx].numpy(), atol=1e-6, rtol=1e-6
                )


if __name__ == '__main__':
    unittest.main()
