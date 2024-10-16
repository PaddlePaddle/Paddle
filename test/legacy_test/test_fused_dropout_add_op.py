#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core
from paddle.incubate.nn.functional import fused_dropout_add
from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd


def paddle_dropout_add(x, y, p=0.5, training=True, mode="upscale_in_train"):
    tmp = paddle.nn.functional.dropout(x, p, training=training, mode=mode)
    return tmp + y


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA ",
)
class TestFusedDropoutAdd(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 1024, 2, 1]
        self.dtype = 'float16'
        self.dropout_rate = 0.5
        self.training = True
        self.mode = "upscale_in_train"
        self.seed = 1027

    def get_paddle_tensor(self):
        tmp = paddle.randn(self.shape, self.dtype)
        tmp.stop_gradient = False
        return tmp

    def get_forward_backward(self, dropout_add, seed):
        paddle.disable_static()
        paddle.seed(seed)
        count = 3
        data = []
        fw = []
        bw = []
        for _ in range(count):
            data.append(self.get_paddle_tensor())

        out = data[0]
        for i in range(1, count):
            out = dropout_add(
                out,
                data[i],
                p=self.dropout_rate,
                training=self.training,
                mode=self.mode,
            )
            fw.append(out)
        out_g = paddle.randn(self.shape, self.dtype)
        paddle.autograd.backward([out], [out_g], True)
        for i in range(count):
            bw.append(data[i].grad)
        return fw, bw

    def test_fused_dropout_add(self):
        p_fw, p_bw = self.get_forward_backward(
            paddle_dropout_add, seed=self.seed
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_dropout_add, seed=self.seed
        )
        for i in range(len(p_fw)):
            np.testing.assert_allclose(
                p_fw[i].numpy(), f_fw[i].numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                p_bw[i].numpy(), f_bw[i].numpy(), rtol=1e-05
            )


def create_test_class(parent, dtype, mode, training, p, seed):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestFusedDropoutAddCase(parent):
        def setUp(self):
            self.shape = (2, 1024, 1, 1)
            self.dtype = dtype
            self.dropout_rate = p
            self.training = training
            self.mode = mode
            self.seed = seed

    cls_name = f"{parent.__name__}_{dtype}_{mode}_{training}_{p}_{seed}"
    TestFusedDropoutAddCase.__name__ = cls_name
    globals()[cls_name] = TestFusedDropoutAddCase


for dtype in ["float64", "float32", "float16"]:
    for mode in ["upscale_in_train", "downscale_in_infer"]:
        for p in [0.0, 0.5, 0.9, 1.0]:
            for training in [True, False]:
                for seed in [0, 1024]:
                    create_test_class(
                        TestFusedDropoutAdd, dtype, mode, training, p, seed
                    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)
class TestFusedDropoutAddStatic(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 80, 8, 2)
        self.dtype = 'float16'

    def test_static_op(self):
        paddle.disable_static()
        paddle.seed(312)
        x_data = np.random.random(self.shape)
        y_data = np.random.random(self.shape)
        x = paddle.to_tensor(
            x_data, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        y = paddle.to_tensor(
            y_data, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        out = fused_dropout_add(x, y, p=0.5, training=True)
        paddle.enable_static()
        paddle.seed(312)

        with paddle.static.program_guard(paddle.static.Program()):
            xs = paddle.static.data(
                name="xs", shape=self.shape, dtype=self.dtype
            )
            ys = paddle.static.data(
                name="ys", shape=self.shape, dtype=self.dtype
            )

            outs = fused_dropout_add(xs, ys, p=0.5, training=True)

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "xs": x_data.astype('float16'),
                    "ys": y_data.astype('float16'),
                },
                fetch_list=[outs],
            )
            np.testing.assert_allclose(out_s[0], out)

    def test_fused_dropout_add_layer(self):
        x = paddle.randn(self.shape, self.dtype)
        y = paddle.randn(self.shape, self.dtype)
        fused_d_a = FusedDropoutAdd(p=0.5)
        d = paddle.nn.Dropout(p=0.5)
        print(d.extra_repr())
        paddle.seed(2048)
        fused_out = fused_d_a(x, y)
        paddle.seed(2048)
        out = d(x) + y
        np.testing.assert_allclose(fused_out, out)

    def test_assert(self):
        def check_raise():
            x = paddle.randn(self.shape, self.dtype)
            y = paddle.randn(self.shape, self.dtype)
            fused_d_a = FusedDropoutAdd(p=-1)
            fused_out = fused_d_a(x, y)

        self.assertRaises(ValueError, check_raise)


if __name__ == '__main__':
    unittest.main()
