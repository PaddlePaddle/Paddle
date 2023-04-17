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
from amp_base_models import AmpTestBase, build_add_model, build_embedding_model

import paddle
from paddle import fluid
from paddle.fluid import core
from paddle.static import amp

paddle.enable_static()


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


class TestProgramBF16(AmpTestBase):
    def _check_bf16_calls(self, op_stats_dict, expected_bf16_calls):
        for op_type, value in expected_bf16_calls.items():
            self.assertEqual(
                op_stats_dict[op_type].bf16_calls,
                value,
                f"The number of bf16 calls of operator < {op_type} > is expected to be {value}, but recieved {op_stats_dict[op_type].bf16_calls}.",
            )

    def test_amp_bf16_o1(self):
        main_program, startup_program = build_embedding_model(
            True, "bfloat16", "O1"
        )
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)
        expected_bf16_calls = {
            "matmul_v2": 1,
            "elementwise_add": 1,
            "dropout": 1,
            "lookup_table_v2": 0,
            "squared_l2_norm": 0,
            "adamw": 0,
        }
        self._check_bf16_calls(op_stats_list[0], expected_bf16_calls)

    def test_amp_bf16_o2(self):
        main_program, startup_program = build_embedding_model(
            True, "bfloat16", "O2"
        )
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)
        expected_bf16_calls = {
            "matmul_v2": 1,
            "elementwise_add": 1,
            "dropout": 1,
            "lookup_table_v2": 0,
            "squared_l2_norm": 2,
            "adamw": 2,
        }
        self._check_bf16_calls(op_stats_list[0], expected_bf16_calls)


class TestStaticBF16(AmpTestBase):
    def _generate_feed_x(self):
        x = np.random.random(size=[16, 16]).astype("float32")
        x_bf16 = convert_float_to_uint16(x)
        x_fp32 = convert_uint16_to_float(x_bf16)
        return x_fp32, x_bf16

    def test_compare_o1_o2(self):
        def _run_o1(place, exe, x_np, max_iters):
            (
                main_program,
                startup_program,
                optimizer,
                feed_vars,
                fetch_vars,
            ) = build_add_model(True, "bfloat16", "O1")

            losses = []
            scope = paddle.static.Scope()
            with paddle.static.scope_guard(scope):
                exe.run(startup_program)
                for iter_id in range(max_iters):
                    results = exe.run(
                        program=main_program,
                        feed={feed_vars[0].name: x_np},
                        fetch_list=fetch_vars,
                    )
                    print(f"-- [BF16 O1] iter={iter_id}, loss={results[0]}")
                    losses.append(results[0])
            return losses

        def _run_o2(place, exe, x_np, max_iters):
            (
                main_program,
                startup_program,
                optimizer,
                feed_vars,
                fetch_vars,
            ) = build_add_model(True, "bfloat16", "O2")

            losses = []
            scope = paddle.static.Scope()
            with paddle.static.scope_guard(scope):
                exe.run(startup_program)
                optimizer.amp_init(place)
                for iter_id in range(max_iters):
                    results = exe.run(
                        program=main_program,
                        feed={feed_vars[0].name: x_np},
                        fetch_list=fetch_vars,
                    )
                    print(f"-- [BF16 O2] iter={iter_id}, loss={results[0]}")
                    losses.append(results[0])
            return losses

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)

        max_iters = 2
        x_fp32, x_bf16 = self._generate_feed_x()
        losses_o1 = _run_o1(place, exe, x_fp32, max_iters)
        losses_o2 = _run_o2(place, exe, x_bf16, max_iters)


if __name__ == '__main__':
    unittest.main()
