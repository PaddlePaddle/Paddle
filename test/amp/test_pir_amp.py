# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.amp.auto_cast import _update_list
from paddle.base import core


class TestAMPState(unittest.TestCase):
    def test_pir_amp_state(self):
        with paddle.pir_utils.IrGuard():
            amp_state = core._get_amp_state()
            amp_state._use_promote = True
            amp_state._amp_level = core.AmpLevel.O2
            amp_state._amp_dtype = 'float16'
            np.testing.assert_equal(core._get_amp_state()._use_promote, True)
            np.testing.assert_equal(
                core._get_amp_state()._amp_level, core.AmpLevel.O2
            )
            np.testing.assert_equal(core._get_amp_state()._amp_dtype, 'float16')
            amp_state._use_promote = False
            amp_state._amp_level = core.AmpLevel.O0
            amp_state._amp_dtype = 'float32'


class TestPirAMPProgram(unittest.TestCase):
    def test_linear_amp_o1(self):
        if not core.is_compiled_with_cuda():
            return
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [3, 4], 'float32')
                linear = paddle.nn.Linear(4, 5)

                amp_state = core._get_amp_state()
                amp_state._use_promote = True
                amp_state._amp_level = core.AmpLevel.O1
                amp_state._amp_dtype = 'float16'
                (
                    original_white_list,
                    original_black_list,
                ) = core._get_amp_op_list()
                _white_list, _black_list = _update_list(
                    None, None, 'O1', 'float16'
                )
                core._set_amp_op_list(_white_list, _black_list)

                out1 = linear(x)
                out2 = paddle.mean(out1)

            cast_op_count = 0
            for op in main.global_block().ops:
                if op.name() == 'pd_op.cast':
                    cast_op_count += 1
            np.testing.assert_equal(out1.dtype, core.DataType.FLOAT32)
            np.testing.assert_equal(out2.dtype, core.DataType.FLOAT32)
            np.testing.assert_equal(cast_op_count, 3)

            amp_state._use_promote = False
            amp_state._amp_level = core.AmpLevel.O0
            amp_state._amp_dtype = 'float32'
            core._set_amp_op_list(original_white_list, original_black_list)
            _white_list, _black_list = core._get_amp_op_list()
            np.testing.assert_equal(len(_white_list), 0)
            np.testing.assert_equal(len(_black_list), 0)


if __name__ == '__main__':
    unittest.main()
