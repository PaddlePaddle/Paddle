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

import paddle


class TestTensorChecker(unittest.TestCase):
    def get_num_inf(self, e):
        num_nan = 0
        num_inf = 0
        # Cannot catch the log in CUDA kernel.
        err_str_list = (
            str(e)
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
            .split(" ")
        )
        for err_str in err_str_list:
            if "num_nan" in err_str:
                num_nan = int(err_str.split("=")[1])
            elif "num_inf" in err_str:
                num_inf = int(err_str.split("=")[1])
        print(
            "[CHECK_NAN_INF_AND_ABORT] num_nan={}, num_inf={}".format(
                num_nan, num_inf
            )
        )
        return num_inf

    def generate_num_inf(self):
        num_inf = 0
        num_nan = 0
        # check op list
        x = paddle.to_tensor(
            [1, 0, 1],
            place=paddle.CPUPlace(),
            dtype='float32',
            stop_gradient=False,
        )
        y = paddle.to_tensor(
            [0.2, -1, 0.5], place=paddle.CPUPlace(), dtype='float32'
        )
        try:
            res = paddle.pow(x, y)
            res = paddle.divide(y, x)
        except Exception as e:
            num_inf = self.get_num_inf(e)
        return num_inf

    def test_check_op_list(self):
        checker_config = paddle.amp.debugging.TensorCheckerConfig(
            enable=True,
            debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF_AND_ABORT,
            skipped_op_list=["elementwise_div"],
        )

        x = paddle.to_tensor(
            [0, 0, 0],
            place=paddle.CPUPlace(),
            dtype='float32',
            stop_gradient=False,
        )
        y = paddle.to_tensor(
            [0.2, -1, 0.5], place=paddle.CPUPlace(), dtype='float32'
        )
        paddle.amp.debugging.enable_tensor_checker(checker_config)
        try:
            res = paddle.divide(y, x)
        except Exception as e:
            num_inf = self.get_num_inf(e)
            self.assertEqual(0, num_inf)

    def test_tensor_checker(self):
        def _assert_flag(value):
            flags = ['FLAGS_check_nan_inf', 'FLAGS_check_nan_inf_level']
            res = paddle.get_flags(flags)
            assert res["FLAGS_check_nan_inf"] == value

        paddle.set_flags({"FLAGS_check_nan_inf": 0})
        paddle.seed(102)
        checker_config = paddle.amp.debugging.TensorCheckerConfig(
            enable=True,
            debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF_AND_ABORT,
            checked_op_list=["elementwise_pow"],
            skipped_op_list=["elementwise_div"],
            debug_step=[0, 3],
        )
        # check seed
        self.assertEqual(checker_config.initial_seed, 102)
        self.assertEqual(checker_config.seed, 102)
        _assert_flag(False)
        for index in range(5):
            paddle.amp.debugging.enable_tensor_checker(checker_config)
            if index <= 2:
                _assert_flag(True)
                self.assertEqual(
                    index + 1,
                    paddle.amp.debugging.TensorCheckerConfig.current_step_id,
                )
                self.assertEqual(1, self.generate_num_inf())
            else:
                self.assertEqual(
                    3, paddle.amp.debugging.TensorCheckerConfig.current_step_id
                )
                _assert_flag(False)
                self.assertEqual(0, self.generate_num_inf())

            paddle.amp.debugging.disable_tensor_checker()
            _assert_flag(False)


if __name__ == '__main__':
    unittest.main()
