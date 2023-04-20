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

import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.fluid import core

_fixed_add_param = np.random.random(size=[16, 16]).astype("float32")


def _build_optimizer(
    use_amp,
    amp_dtype="float16",
    amp_level="O1",
    amp_lists=None,
    use_grad_clip=False,
):
    if use_grad_clip:
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    else:
        grad_clip = None
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.01,
        grad_clip=grad_clip,
        beta1=0.78,
        beta2=0.836,
        epsilon=1e-4,
        weight_decay=0.01,
        multi_precision=True,
    )
    if use_amp:
        optimizer = paddle.static.amp.decorate(
            optimizer, amp_lists, level=amp_level, dtype=amp_dtype
        )
    return optimizer


class SimpleAddNet(nn.Layer):
    def __init__(self, dtype):
        super().__init__()
        global _fixed_add_param
        self.weight = paddle.create_parameter(
            name="add_w",
            shape=[16, 16],
            dtype=dtype,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(_fixed_add_param)
            ),
        )

    def forward(self, x):
        return x + self.weight


def build_add_model(use_amp, amp_dtype="float16", amp_level="O1"):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            x_dtype = "float32"
            if use_amp and amp_level == "O2":
                if amp_dtype == "bfloat16":
                    x_dtype = "uint16"
                elif amp_dtype == "float16":
                    x_dtype = "float16"
            model = SimpleAddNet(x_dtype)
            x = paddle.static.data(name='input', shape=[16, 16], dtype=x_dtype)
            out = model(x)
            loss = paddle.mean(out)

            if use_amp:
                amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
                    custom_white_list=["elementwise_add"],
                    custom_black_list=["reduce_mean"],
                    dtype=amp_dtype,
                )
            else:
                amp_lists = None
            optimizer = _build_optimizer(
                use_amp, amp_dtype, amp_level, amp_lists
            )
            optimizer.minimize(loss)
    feed_vars = [x]
    fetch_vars = [loss]
    return main_program, startup_program, optimizer, feed_vars, fetch_vars


class SimpleConvNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3)
        self.linear = nn.Linear(in_features=6, out_features=10)

    def forward(self, x):
        out = self.conv(x)
        out = nn.functional.relu(out)
        out = self.linear(out)
        out = nn.functional.softmax(out)
        return out


def build_conv_model(use_amp, amp_dtype="float16", amp_level="O1"):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleConvNet()
            x = paddle.static.data(
                name='input', shape=[None, 1, 28, 28], dtype='float32'
            )
            out = model(x)
            loss = paddle.mean(out)
            optimizer = _build_optimizer(use_amp, amp_dtype, amp_level)
            optimizer.minimize(loss)
    return main_program, startup_program


class SimpleEmbeddingNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.vocab_size = 128
        self.hidden_size = 16
        self.vocab_size = 128
        self.hidden_size = 16
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.linear = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        out = self.embedding(x)
        scale = paddle.full(shape=[1], fill_value=2, dtype="int64")
        out = paddle.multiply(out, scale.astype("float32"))
        out = self.linear(out)
        out = nn.functional.dropout(out, p=0.2)
        return out


def build_embedding_model(use_amp, amp_dtype="float16", amp_level="O1"):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleEmbeddingNet()
            x = paddle.static.data(name='x', shape=[None, 32], dtype='int64')
            out = model(x)
            loss = paddle.mean(out)
            optimizer = _build_optimizer(
                use_amp, amp_dtype, amp_level, None, True
            )
            optimizer.minimize(loss)
    return main_program, startup_program


class SimpleWhileNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(16, 10)

    def forward(self, x):
        def cond(i, loop_len, x, result):
            return i < loop_len

        def body(i, loop_len, x, result):
            result = self.linear(x)
            paddle.increment(i)
            return [i, loop_len, x, result]

        i = paddle.zeros(shape=[1], dtype='int64')
        loop_len = paddle.ones(shape=[1], dtype='int64')
        result = paddle.zeros(
            shape=x.shape[:-1] + self.linear.weight.shape[-1:], dtype="float32"
        )
        result.stop_gradient = False
        _, _, _, results = paddle.static.nn.while_loop(
            cond, body, [i, loop_len, x, result]
        )
        return results


def build_while_model():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleWhileNet()
            x = paddle.static.data(name='x', shape=[32, 16], dtype='float32')
            out = model(x)
            loss = paddle.mean(out)
    return main_program, startup_program


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not complied with CUDA and not support amp.",
)
class AmpTestBase(unittest.TestCase):
    def setUp(self):
        self.amp_dtype = None
        self.amp_level = None
