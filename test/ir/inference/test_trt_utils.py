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

paddle.enable_static()


class TestTRTUtils(unittest.TestCase):
    def test_collect_shape(self):
        paddle.framework.set_flags({"FLAGS_enable_collect_shape": True})
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.static.data('x', [4, 4], dtype='float32')
                y = paddle.static.data('y', [4, 4], dtype='float32')
                divide_out = paddle.divide(x, y)
                sum_out = paddle.sum(divide_out)
                exe = paddle.static.Executor()
                x_feed = np.ones([4, 4], dtype=np.float32) * 10
                y_feed = np.ones([4, 4], dtype=np.float32) * 2
            (sum_value,) = exe.run(
                main_program,
                feed={'x': x_feed, 'y': y_feed},
                fetch_list=[sum_out],
            )

            # program, _, _ = exe._executor_cache.get_pir_program_and_executor(
            #     main_program,
            #     feed={'x': x_feed, 'y': y_feed},,
            #     fetch_list=[sum_out],
            #     feed_var_name='feed',
            #     fetch_var_name='fetch',
            #     place=paddle.framework._current_expected_place_(),
            #     scope=global_scope(),
            #     plan=None,
            # )

            # divide_out_min_shape = paddle.base.core.get_value_shape_range_info(
            #     divide_out, False, paddle.base.core.ShapeMode.kMIN
            # )
            # divide_out_max_shape = paddle.base.core.get_value_shape_range_info(
            #     divide_out, False, paddle.base.core.ShapeMode.kMAX
            # )
            # divide_out_opt_shape = paddle.base.core.get_value_shape_range_info(
            #     divide_out, False, paddle.base.core.ShapeMode.kOPT
            # )
            # self.assertEqual(len(divide_out_min_shape), 2)
            # self.assertEqual(len(divide_out_max_shape), 2)
            # self.assertEqual(len(divide_out_opt_shape), 2)
            # self.assertEqual(divide_out_min_shape[0], 4)
            # self.assertEqual(divide_out_max_shape[0], 4)
            # self.assertEqual(divide_out_opt_shape[0], 4)
        paddle.framework.set_flags({"FLAGS_enable_collect_shape": False})


if __name__ == "__main__":
    unittest.main()
