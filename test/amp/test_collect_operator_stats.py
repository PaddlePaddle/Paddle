# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

sys.path.append("../../amp")
from amp_base_models import build_while_model

import paddle


class TestOpStatsEager(unittest.TestCase):
    def _check_result(self, dtype):
        # Returned the dict.
        op_list = paddle.base.core.get_low_precision_op_list()

        self.assertTrue('elementwise_add' in op_list)
        self.assertTrue('conv2d' in op_list)

        conv2d_called = op_list['conv2d'].split(',')
        add_called = op_list['elementwise_add'].split(',')
        add_num = 0
        conv_num = 0
        for i in range(4):
            add_num += int(add_called[i])
            conv_num += int(add_called[i])

        self.assertTrue(conv_num == 1)
        self.assertTrue(add_num == 1)

        if dtype == paddle.float16:
            self.assertTrue(int(conv2d_called[0]) == 1)
            self.assertTrue(int(add_called[0]) == 1)

    def test_enable_disable(self):
        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])

        paddle.amp.debugging.enable_operator_stats_collection()
        # amp list conv2d, elementwise_add, cast (transfer_dtype)
        with paddle.amp.auto_cast(enable=True, level='O2'):
            out = conv(x)
        # Print to the standard output.
        paddle.amp.debugging.disable_operator_stats_collection()

        self._check_result(dtype=out.dtype)

    def test_context(self):
        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])

        with paddle.amp.debugging.collect_operator_stats():
            # amp list conv2d, elementwise_add, cast (transfer_dtype)
            with paddle.amp.auto_cast(enable=True, level='O2'):
                out = conv(x)

        self._check_result(dtype=out.dtype)


class TestOpStatsPir(unittest.TestCase):
    def _check_result(self, dtype):
        # Returned the dict.
        op_list = paddle.base.core.get_low_precision_op_list()

        self.assertTrue('pd_op.add' in op_list)
        self.assertTrue('pd_op.conv2d' in op_list)

        conv2d_called = op_list['pd_op.conv2d'].split(',')
        add_called = op_list['pd_op.add'].split(',')
        add_num = 0
        conv_num = 0
        for i in range(4):
            add_num += int(add_called[i])
            conv_num += int(add_called[i])

        self.assertTrue(conv_num == 1)
        self.assertTrue(add_num == 1)

        if dtype == paddle.float16:
            self.assertTrue(int(conv2d_called[0]) == 1)
            self.assertTrue(int(add_called[0]) == 1)

    def test_enable_disable(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle.set_flags({"FLAGS_pir_apply_inplace_pass": 0})
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                conv = paddle.nn.Conv2D(3, 2, 3)
                x = paddle.static.data('x', [10, 3, 32, 32], 'float32')

                with paddle.amp.auto_cast(enable=True, level='O2'):
                    out = conv(x)

                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(startup)
                paddle.amp.debugging.enable_operator_stats_collection()
                exe.run(
                    main,
                    feed={
                        'x': np.random.random([10, 3, 32, 32]).astype(
                            'float32'
                        ),
                    },
                    fetch_list=[out],
                )
                paddle.amp.debugging.disable_operator_stats_collection()
                self._check_result(dtype=out.dtype)

    def test_context(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle.set_flags({"FLAGS_pir_apply_inplace_pass": 0})
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                conv = paddle.nn.Conv2D(3, 2, 3)
                x = paddle.static.data('x', [10, 3, 32, 32], 'float32')
                with paddle.amp.auto_cast(enable=True, level='O2'):
                    out = conv(x)

            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(startup)
            with paddle.amp.debugging.collect_operator_stats():
                exe.run(
                    main,
                    feed={
                        'x': np.random.random([10, 3, 32, 32]).astype(
                            'float32'
                        ),
                    },
                    fetch_list=[out],
                )
            self._check_result(dtype=out.dtype)


class TestOpStatsStatic(unittest.TestCase):
    def test_while_op(self):
        paddle.enable_static()
        main_program, startup_program = build_while_model()
        if paddle.framework.use_pir_api():
            self.assertEqual(main_program.num_blocks, 1)
        else:
            self.assertEqual(main_program.num_blocks, 2)

            paddle.static.amp.debugging.collect_operator_stats(
                program=main_program, print_subblocks=True
            )
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
