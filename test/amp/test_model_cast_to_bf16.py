# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import struct
import unittest

import numpy as np

import paddle
from paddle import fluid, nn
from paddle.fluid import core
from paddle.static import amp

paddle.enable_static()


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


cutf = convert_uint16_to_float


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestModelCastBF16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    @contextlib.contextmanager
    def static_graph(self):
        with self.scope_prog_guard():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield

    def get_static_graph_result(
        self, feed, fetch_list, amp_fun, with_lod=False, startup_prog=None
    ):
        exe = fluid.Executor(core.CPUPlace())
        exe.run(
            fluid.default_startup_program()
            if startup_prog is None
            else startup_prog
        )
        prog = fluid.default_main_program()
        if amp_fun is not None:
            if startup_prog is not None:
                amp_fun(prog, startup_prog)
            else:
                amp_fun(prog)
        return exe.run(
            prog, feed=feed, fetch_list=fetch_list, return_numpy=(not with_lod)
        )

    def _graph_common(self, _amp_fun, startup_prog=None):
        size = 3
        n = np.ones([size, size], dtype='float32') * 3.2
        nn = np.ones([size, size], dtype='float32') * -2.7

        n_bf16 = amp.bf16.convert_float_to_uint16(n)
        nn_bf16 = amp.bf16.convert_float_to_uint16(nn)

        with self.static_graph():
            t_bf16 = paddle.static.data(
                name='t_bf16', shape=[-1, size, size], dtype='int32'
            )
            t_bf16.desc.set_need_check_feed(False)
            tt_bf16 = paddle.static.data(
                name='tt_bf16', shape=[-1, size, size], dtype='int32'
            )
            tt_bf16.desc.set_need_check_feed(False)
            t = paddle.static.data(
                name='t', shape=[-1, size, size], dtype='float32'
            )
            t.desc.set_need_check_feed(False)
            tt = paddle.static.data(
                name='tt', shape=[-1, size, size], dtype='float32'
            )
            tt.desc.set_need_check_feed(False)

            ret = paddle.add(t, tt)
            ret = paddle.multiply(ret, t)
            ret = paddle.reshape(ret, [0, 0])

            with amp.bf16.bf16_guard():
                ret_bf16 = paddle.add(t_bf16, tt_bf16)
                ret_bf16 = paddle.multiply(ret_bf16, t_bf16)
                ret_bf16 = paddle.reshape(ret_bf16, [0, 0])

            with amp.bf16.bf16_guard():
                ret_fp32bf16 = paddle.add(t, tt)
                ret_fp32bf16 = paddle.multiply(ret_fp32bf16, t)
                ret_fp32bf16 = paddle.reshape(ret_fp32bf16, [0, 0])

            (
                static_ret_bf16,
                static_ret,
                ret_fp32bf16,
            ) = self.get_static_graph_result(
                feed={
                    't': n,
                    'tt': nn,
                    't_bf16': n_bf16,
                    'tt_bf16': nn_bf16,
                },
                fetch_list=[ret_bf16, ret, ret_fp32bf16],
                amp_fun=_amp_fun,
                startup_prog=startup_prog,
            )

        np.testing.assert_allclose(
            cutf(static_ret_bf16), cutf(static_ret), rtol=0.01
        )
        np.testing.assert_allclose(
            cutf(static_ret_bf16), cutf(ret_fp32bf16), rtol=0.01
        )

        with self.static_graph():
            t = paddle.static.data(
                name='t', shape=[-1, size, size], dtype='float32'
            )
            t.desc.set_need_check_feed(False)
            tt = paddle.static.data(
                name='tt', shape=[-1, size, size], dtype='float32'
            )
            tt.desc.set_need_check_feed(False)

            with amp.bf16.bf16_guard():
                ret = paddle.add(t, tt)
                ret = paddle.reshape(ret, [0, 0])
                ret = paddle.nn.functional.elu(ret)
                ret = paddle.multiply(ret, t)
            ret = paddle.add(ret, tt)

            static_ret_bf16 = self.get_static_graph_result(
                feed={'t': n, 'tt': nn},
                fetch_list=[ret],
                amp_fun=_amp_fun,
                startup_prog=startup_prog,
            )
        self.assertTrue(
            static_ret_bf16, np.ones([size, size], dtype='float32') * -1.1
        )

    def test_graph_rewrite(self):
        self._graph_common(
            lambda prog: amp.bf16.rewrite_program_bf16(
                prog,
                amp.bf16.AutoMixedPrecisionListsBF16(
                    custom_bf16_list={'elementwise_add'},
                    custom_fp32_varnames={'elementwise_add_0.tmp_0'},
                ),
            )
        )

    def test_graph_cast(self):
        self._graph_common(
            lambda prog, startup_prog: amp.bf16.cast_model_to_bf16(
                prog,
                startup_prog,
                amp.bf16.AutoMixedPrecisionListsBF16(
                    custom_bf16_list={'elementwise_add'},
                    custom_fp32_list={'elementwise_mul'},
                ),
                use_bf16_guard=True,
            ),
            startup_prog=fluid.default_startup_program(),
        )


class SimpleNet(nn.Layer):
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


def _build_model(use_amp, amp_dtype="float16", amp_level="O1"):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startup_program):
            model = SimpleNet()
            x = paddle.static.data(name='x', shape=[None, 32], dtype='int64')
            out = model(x)
            loss = paddle.mean(out)
            optimizer = paddle.optimizer.AdamW(
                learning_rate=0.01,
                grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
                beta1=0.78,
                beta2=0.836,
                epsilon=1e-4,
                weight_decay=0.01,
                multi_precision=True,
            )
            if use_amp:
                optimizer = paddle.static.amp.amp_decorate(
                    optimizer, level=amp_level, dtype=amp_dtype
                )
            optimizer.minimize(loss)
    return main_program, startup_program


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestProgramBF16(unittest.TestCase):
    def test_amp_bf16_o1(self):
        main_program, startup_program = _build_model(True, "bfloat16", "O1")
        print(main_program)
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)

        op_stats_dict = op_stats_list[0]
        expected_bf16_calls = {
            "matmul_v2": 1,
            "elementwise_add": 1,
            "dropout": 1,
            "lookup_table_v2": 0,
            "squared_l2_norm": 0,
            "adamw": 0,
        }
        for op_type, value in expected_bf16_calls.items():
            self.assertEqual(
                op_stats_dict[op_type].bf16_calls,
                value,
                f"The number of bf16 calls of operator < {op_type} > is expected to be {value}, but recieved {op_stats_dict[op_type].bf16_calls}.",
            )

    def test_amp_bf16_o2(self):
        main_program, startup_program = _build_model(True, "bfloat16", "O2")
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)

        op_stats_dict = op_stats_list[0]
        expected_bf16_calls = {
            "matmul_v2": 1,
            "elementwise_add": 1,
            "dropout": 1,
            "lookup_table_v2": 0,
            "squared_l2_norm": 2,
            "adamw": 2,
        }
        for op_type, value in expected_bf16_calls.items():
            self.assertEqual(
                op_stats_dict[op_type].bf16_calls,
                value,
                f"The number of bf16 calls of operator < {op_type} > is expected to be {value}, but recieved {op_stats_dict[op_type].bf16_calls}.",
            )


if __name__ == '__main__':
    unittest.main()
