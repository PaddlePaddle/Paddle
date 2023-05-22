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


import random
import unittest

import numpy as np
from amp_base_models import AmpTestBase, _build_optimizer

import paddle
from paddle import nn

paddle.enable_static()

_fixed_param = np.random.random(size=[64, 64]).astype("float32")


class SimpleUnittedEmbeddingNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.vocab_size = 64
        self.hidden_size = 64
        global _fixed_param

        self.param_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(_fixed_param)
        )
        self.embedding = nn.Embedding(
            self.vocab_size, self.hidden_size, weight_attr=self.param_attr
        )
        self.linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            weight_attr=self.param_attr,
        )

    def forward(self, x):
        out = self.embedding(x)
        scale = paddle.full(shape=[1], fill_value=2, dtype="int64")
        out = paddle.multiply(out, scale.astype("float32"))
        out = self.linear(out)
        out = nn.functional.dropout(out, p=0.2)
        return out


def build_unitted_embedding_model(
    use_amp,
    amp_dtype="float16",
    amp_level="O1",
    use_promote=False,
):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleUnittedEmbeddingNet()
            x = paddle.static.data(name='x', shape=[None, 32], dtype='int64')
            out = model(x)
            loss = paddle.mean(out)
            if use_amp:
                amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
                    custom_white_list=["elementwise_mul"],
                    custom_black_list=["reduce_mean"],
                    dtype=amp_dtype,
                )
            else:
                amp_lists = None
            optimizer = _build_optimizer(
                use_amp,
                amp_dtype,
                amp_level,
                amp_lists,
                True,
                use_promote=use_promote,
            )
            optimizer.minimize(loss)

    feed_vars = [x]
    fetch_vars = [loss]
    return main_program, startup_program, optimizer, feed_vars, fetch_vars


class TestUnittedEmbedding(AmpTestBase):
    def _generate_feed_x(self):
        seed = 0
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        x = np.random.randint(1, 64, size=[1, 32]).astype("int64")
        return x

    def test_compare_o1_and_o2_master_grad(self):
        def _run(place, exe, x_np, max_iters, level):
            (
                main_program,
                startup_program,
                optimizer,
                feed_vars,
                fetch_vars,
            ) = build_unitted_embedding_model(
                True,
                "float16",
                level,
            )

            seed = 0
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            losses = self.run_program(
                main_program,
                startup_program,
                optimizer,
                feed_vars,
                fetch_vars,
                place,
                exe,
                x_np,
                max_iters,
                "float16",
                level,
            )
            return losses

        max_iters = 5
        x = self._generate_feed_x()
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        losses_o2 = _run(place, exe, x, max_iters, 'O2')


if __name__ == "__main__":
    unittest.main()
