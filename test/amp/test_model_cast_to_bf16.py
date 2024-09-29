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
import unittest

import numpy as np
from amp_base_models import (
    AmpTestBase,
    build_add_model,
    build_embedding_model,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

import paddle
from paddle import base
from paddle.base import core
from paddle.static import amp

paddle.enable_static()

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
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                yield

    def get_static_graph_result(
        self, feed, fetch_list, amp_fun, with_lod=False, startup_prog=None
    ):
        exe = base.Executor(core.CPUPlace())
        exe.run(
            base.default_startup_program()
            if startup_prog is None
            else startup_prog
        )
        prog = base.default_main_program()
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
        with paddle.pir_utils.OldIrGuard():
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
        with paddle.pir_utils.OldIrGuard():
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
                startup_prog=base.default_startup_program(),
            )


@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestProgramBF16(AmpTestBase):
    def _check_optimizer(self, program, expected_num_mp):
        optimizers = []
        for block in program.blocks:
            for op in block.ops:
                if "Param" in op.input_names and "Grad" in op.input_names:
                    optimizers.append(op)

        actual_num_mp = 0
        for op in optimizers:
            if op.has_attr("multi_precision") and op.attr("multi_precision"):
                actual_num_mp += 1
        self.assertEqual(
            actual_num_mp,
            expected_num_mp,
            f"The number of optimizers with multi_precision = True is expected to be {expected_num_mp}, but received {actual_num_mp}.",
        )

    def test_amp_bf16_o1(self):
        with paddle.pir_utils.OldIrGuard():
            main_program, startup_program, _, _, _ = build_embedding_model(
                True, "bfloat16", "O1"
            )
            self.assertEqual(main_program.num_blocks, 1)
            self._check_optimizer(main_program, 0)

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
            self._check_op_calls(op_stats_list[0], expected_bf16_calls)

    def test_amp_bf16_o2(self):
        with paddle.pir_utils.OldIrGuard():
            main_program, startup_program, _, _, _ = build_embedding_model(
                True, "bfloat16", "O2"
            )
            self.assertEqual(main_program.num_blocks, 1)

            amp.debugging.collect_operator_stats(main_program)
            op_stats_list = amp.debugging._get_op_stats_list(main_program)
            expected_fp32_calls = {"lookup_table_v2": 1}
            expected_bf16_calls = {
                "matmul_v2": 1,
                "elementwise_add": 1,
                "dropout": 1,
                "lookup_table_v2": 0,
                "squared_l2_norm": 3,
                "adamw": 3,
            }
            self._check_optimizer(
                main_program,
                expected_bf16_calls["matmul_v2"]
                + expected_bf16_calls["elementwise_add"]
                + expected_fp32_calls["lookup_table_v2"],
            )
            self._check_op_calls(op_stats_list[0], expected_bf16_calls)


@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestStaticBF16(AmpTestBase):
    def _generate_feed_x(self):
        x = np.random.random(size=[16, 16]).astype("float32")
        x_bf16 = convert_float_to_uint16(x)
        x_fp32 = convert_uint16_to_float(x_bf16)
        return x_fp32, x_bf16

    def test_compare_o1_o2(self):
        with paddle.pir_utils.OldIrGuard():

            def _run(place, exe, x_np, max_iters, level):
                (
                    main_program,
                    startup_program,
                    optimizer,
                    feed_vars,
                    fetch_vars,
                ) = build_add_model(True, "bfloat16", level)

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
                    "bfloat16",
                    level,
                )
                return losses

            max_iters = 2
            x_fp32, x_bf16 = self._generate_feed_x()
            if paddle.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            elif paddle.is_compiled_with_xpu():
                place = paddle.device.XPUPlace(0)
            else:
                raise ValueError("Only support CUDA or XPU Place.")
            exe = paddle.static.Executor(place)
            losses_o1 = _run(place, exe, x_fp32, max_iters, 'O1')
            losses_o2 = _run(place, exe, x_bf16, max_iters, 'O2')

            self.assertEqual(
                losses_o1,
                losses_o2,
                f"loss of o1 and o2 should be equal, but received loss o1: {losses_o1}, loss o2: {losses_o2}",
            )


if __name__ == '__main__':
    unittest.main()
