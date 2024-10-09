#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import in_pir_mode
from paddle.nn import functional
from paddle.nn.functional.input import embedding_renorm_


class EmbeddingStatic(unittest.TestCase):
    def test_1(self):
        prog = base.Program()
        with base.program_guard(prog):

            def test_bad_x():
                initializer = paddle.nn.initializer.Assign(
                    np.random.random(size=(128, 100))
                )

                param_attr = base.ParamAttr(
                    name="emb_weight",
                    learning_rate=0.5,
                    initializer=initializer,
                    trainable=True,
                )

                if in_pir_mode():
                    weight = paddle.pir.core.create_parameter(
                        shape=(128, 100),
                        dtype="float32",
                        **param_attr._to_kwargs(with_initializer=True),
                    )
                else:
                    weight = prog.global_block().create_parameter(
                        (128, 100), attr=param_attr, dtype="float32"
                    )

                label = paddle.static.data(
                    name="label",
                    shape=[-1, 4],
                    dtype="int64",
                )

                emb = functional.embedding(
                    x=label, weight=weight, sparse=True, name="embedding"
                )

            test_bad_x()

    def test_2(self):
        prog = base.Program()
        with base.program_guard(prog):

            def test_bad_x():
                initializer = paddle.nn.initializer.Assign(
                    np.random.random(size=(128, 100))
                )

                param_attr = base.ParamAttr(
                    name="emb_weight",
                    learning_rate=0.5,
                    initializer=initializer,
                    trainable=True,
                )

                if in_pir_mode():
                    weight = paddle.pir.core.create_parameter(
                        shape=(128, 100),
                        dtype="float32",
                        **param_attr._to_kwargs(with_initializer=True),
                    )
                else:
                    weight = prog.global_block().create_parameter(
                        (128, 100), attr=param_attr, dtype="float32"
                    )

                label = paddle.static.data(
                    name="label",
                    shape=[-1, 4],
                    dtype="int32",
                )

                emb = functional.embedding(
                    x=label,
                    weight=weight,
                    padding_idx=129,
                    sparse=True,
                    name="embedding",
                )

        with self.assertRaises(ValueError):
            test_bad_x()

    def test_3_renorm(self):
        x_np = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int64)
        weight_np = np.random.random((10, 4)).astype(np.float32) * 10
        max_norm = 5.0
        norm_type = 2.0
        y_ref = self.ref_embedding_renorm_(x_np, weight_np, max_norm, norm_type)
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.static.data(name="x", shape=[-1, 3], dtype="int64")
            weight = paddle.static.data(
                name="weight", shape=[10, 4], dtype="float32"
            )
            res = embedding_renorm_(
                x=x, weight=weight, max_norm=max_norm, norm_type=norm_type
            )
            exe = paddle.static.Executor(place)
            res_val = exe.run(
                prog, feed={"x": x_np, "weight": weight_np}, fetch_list=[res]
            )
            paddle_result = res_val[0]
        np.testing.assert_allclose(paddle_result, y_ref, atol=1e-5)

    def ref_embedding_renorm_(self, x, weight, max_norm, norm_type=2.0):
        x = np.reshape(x, (-1,))
        x = np.unique(x)
        x = np.sort(x)
        for i in range(len(x)):
            norm = np.linalg.norm(
                weight[int(x[i])], ord=norm_type, axis=0, keepdims=False
            )
            if norm > max_norm:
                weight[int(x[i])] *= max_norm / (norm + 1e-7)
        return weight


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
