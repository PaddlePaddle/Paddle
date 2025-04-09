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

import copy
import struct
import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.base import core
from paddle.framework import in_dynamic_or_pir_mode


def copy_bits_from_float_to_uint16(f):
    return struct.unpack('<I', struct.pack('<f', f))[0] >> 16


def convert_float_to_uint16(in_list):
    if in_list.dtype == np.float32:
        new_output = []
        for x in np.nditer(in_list):
            new_output.append(np.uint16(copy_bits_from_float_to_uint16(x)))
        new_output = np.reshape(new_output, in_list.shape).view(np.uint16)
        return new_output
    else:
        return in_list


def convert_uint16_to_float(in_list):
    if in_list.dtype == np.uint16:
        in_list = np.asarray(in_list)
        out = np.vectorize(
            lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0],
            otypes=[np.float32],
        )(in_list.flat)
        return np.reshape(out, in_list.shape)
    else:
        return in_list


_fixed_add_param = np.random.random(size=[16, 16]).astype("float32")


def _build_optimizer(
    use_amp,
    amp_dtype="float16",
    amp_level="O1",
    amp_lists=None,
    use_grad_clip=False,
    use_promote=False,
    use_master_grad=False,
    model=None,
):
    if use_grad_clip:
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    else:
        grad_clip = None
    if in_dynamic_or_pir_mode():
        assert model is not None
        parameters = model.parameters()
    else:
        parameters = None
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.01,
        parameters=parameters,
        grad_clip=grad_clip,
        beta1=0.78,
        beta2=0.836,
        epsilon=1e-4,
        weight_decay=0.01,
    )
    if not in_dynamic_or_pir_mode() and use_amp:
        optimizer = paddle.static.amp.decorate(
            optimizer,
            amp_lists,
            level=amp_level,
            dtype=amp_dtype,
            master_grad=use_master_grad,
            use_promote=use_promote,
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


def cast_add_param(amp_dtype):
    global _fixed_add_param
    if amp_dtype == "bfloat16":
        _fixed_add_param_bf16 = convert_float_to_uint16(_fixed_add_param)
        _fixed_add_param = convert_uint16_to_float(_fixed_add_param_bf16)
    else:
        pass


def build_add_model(
    use_amp, amp_dtype="float16", amp_level="O1", use_promote=False
):
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
            cast_add_param(amp_dtype)
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
                use_amp,
                amp_dtype,
                amp_level,
                amp_lists,
                use_promote=use_promote,
            )
            optimizer.minimize(loss)
    feed_vars = [x]
    fetch_vars = [loss]
    return main_program, startup_program, optimizer, feed_vars, fetch_vars


class SimpleConvNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3)
        self.linear = nn.Linear(in_features=96, out_features=4)

    def forward(self, x):
        out = self.conv(x)
        out = nn.functional.relu(out.cast("float32"))
        out = out.flatten(start_axis=1, stop_axis=3)
        out = self.linear(out)
        out = nn.functional.softmax(out)
        return out


def build_conv_model(
    use_amp, amp_dtype="float16", amp_level="O1", use_promote=False
):
    if in_dynamic_or_pir_mode():
        model = SimpleConvNet()
        optimizer = _build_optimizer(use_amp=False, model=model)
        if use_amp and amp_dtype == "float16":
            scaler = paddle.amp.GradScaler(init_loss_scaling=32768.0)
        else:
            scaler = None
        if use_amp and amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                dtype=amp_dtype,
            )
        return model, optimizer, scaler

    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleConvNet()
            x = paddle.static.data(
                name='input', shape=[None, 1, 6, 6], dtype='float32'
            )
            out = model(x)
            loss = paddle.mean(out)
            optimizer = _build_optimizer(
                use_amp, amp_dtype, amp_level, use_promote=use_promote
            )
            optimizer.minimize(loss)
    feed_vars = [x]
    fetch_vars = [loss]
    return main_program, startup_program, optimizer, feed_vars, fetch_vars


class SimpleEmbeddingNet(nn.Layer):
    def __init__(self):
        super().__init__()
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


def build_embedding_model(
    use_amp,
    amp_dtype="float16",
    amp_level="O1",
    use_promote=False,
    use_master_grad=False,
):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleEmbeddingNet()
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
                use_master_grad=use_master_grad,
            )
            optimizer.minimize(loss)

    feed_vars = [x]
    fetch_vars = [loss]
    return main_program, startup_program, optimizer, feed_vars, fetch_vars


class SimpleMLPNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear0 = paddle.nn.Linear(16, 10)
        self.linear1 = paddle.nn.Linear(10, 32)

    def forward(self, x):
        out = self.linear0(x)
        out = nn.functional.relu(out)
        out = self.linear1(out)
        out = nn.functional.relu(out)
        out = nn.functional.dropout(out, p=0.2)
        return out


def build_MLP_model(
    use_amp,
    use_grad_clip=False,
    amp_dtype="float16",
    amp_level="O1",
    use_promote=False,
    use_master_grad=False,
):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleMLPNet()
            x_dtype = "float32"
            if use_amp and amp_level == "O2":
                if amp_dtype == "bfloat16":
                    x_dtype = "uint16"
                elif amp_dtype == "float16":
                    x_dtype = "float16"
            x = paddle.static.data(name='x', shape=[None, 16], dtype=x_dtype)
            out = model(x)
            loss = paddle.mean(out)

            if use_amp:
                amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
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
                use_grad_clip=use_grad_clip,
                use_promote=use_promote,
                use_master_grad=use_master_grad,
            )
            optimizer.minimize(loss)

    feed_vars = [x]
    fetch_vars = [loss]
    return main_program, startup_program, optimizer, feed_vars, fetch_vars


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
    not (core.is_compiled_with_xpu()),
    "core is not compiled with XPU and not support amp.",
)
class AmpTestBase(unittest.TestCase):
    def setUp(self):
        self.amp_dtype = None
        self.amp_level = None

    def _check_op_calls(
        self,
        op_stats_dict,
        expected_bf16_calls={},
        expected_fp16_calls={},
        debug_info=None,
    ):
        def _extract_op_call(op_calls_str, pos):
            return int(copy.copy(op_calls_str).split(",")[pos])

        for op_type, expected_value in expected_bf16_calls.items():
            # print(f"[BF16] op_type={op_type}, value={value}")
            if isinstance(op_stats_dict[op_type], str):
                actual_value = _extract_op_call(op_stats_dict[op_type], 1)
            else:
                actual_value = op_stats_dict[op_type].bf16_calls
            self.assertEqual(
                actual_value,
                expected_value,
                f"[{debug_info}] The number of bf16 calls of operator < {op_type} > is expected to be {expected_value}, but received {actual_value}.",
            )
        for op_type, expected_value in expected_fp16_calls.items():
            # print(f"[FP16] op_type={op_type}, value={value}")
            if isinstance(op_stats_dict[op_type], str):
                actual_value = _extract_op_call(op_stats_dict[op_type], 0)
            else:
                actual_value = op_stats_dict[op_type].fp16_calls
            self.assertEqual(
                actual_value,
                expected_value,
                f"[debug_info] The number of fp16 calls of operator < {op_type} > is expected to be {expected_value}, but received {actual_value}.",
            )

    def run_program(
        self,
        main_program,
        startup_program,
        optimizer,
        feed_vars,
        fetch_vars,
        place,
        exe,
        x_np,
        max_iters,
        dtype,
        level,
    ):
        losses = []
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup_program)
            if level == 'O2':
                optimizer.amp_init(place)
            for iter_id in range(max_iters):
                results = exe.run(
                    program=main_program,
                    feed={feed_vars[0].name: x_np},
                    fetch_list=fetch_vars,
                )
                print(
                    f"-- [AMP {dtype} {level}] iter={iter_id}, loss={results[0]}"
                )
                losses.append(results[0])
        return losses
