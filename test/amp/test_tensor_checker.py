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

import os
import unittest

import paddle


class TestTensorChecker(unittest.TestCase):
    def _parse_num_nan_inf(self, e):
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
        return num_nan, num_inf

    def _generate_num_inf(self, place):
        num_inf = 0
        num_nan = 0
        paddle.set_device(place)
        # check op list
        x = paddle.to_tensor(
            [1, 0, 0],
            dtype='float32',
            stop_gradient=False,
        )
        y = paddle.to_tensor([0, 0, 1], dtype='float32')
        try:
            res = paddle.pow(x, y)
            # test backward
            paddle.autograd.backward([res])
            res = paddle.divide(y, x)
        except Exception as e:
            num_nan, num_inf = self._parse_num_nan_inf(e)
        return num_nan, num_inf

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
            checked_op_list=["elementwise_pow_grad"],
            skipped_op_list=["elementwise_div"],
            debug_step=[0, 3],
        )
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        # check seed
        self.assertEqual(checker_config.initial_seed, 102)
        self.assertEqual(checker_config.seed, 102)
        _assert_flag(False)

        for place in places:
            paddle.amp.debugging.TensorCheckerConfig.current_step_id = 0
            for iter_id in range(5):
                paddle.amp.debugging.enable_tensor_checker(checker_config)
                if iter_id <= 2:
                    _assert_flag(True)
                    self.assertEqual(
                        iter_id + 1,
                        paddle.amp.debugging.TensorCheckerConfig.current_step_id,
                    )
                    num_nan, num_inf = self._generate_num_inf(place)
                    print(
                        f"-- [iter_id={iter_id}, place={place}] num_nan={num_nan}, num_inf={num_inf}"
                    )
                    self.assertEqual(
                        1,
                        num_nan,
                        f"Expected num_nan to be 1, but received {num_nan}, place={place}.",
                    )
                else:
                    self.assertEqual(
                        3,
                        paddle.amp.debugging.TensorCheckerConfig.current_step_id,
                    )
                    _assert_flag(False)
                    num_nan, num_inf = self._generate_num_inf(place)
                    print(
                        f"-- [iter_id={iter_id}, place={place}] num_nan={num_nan}, num_inf={num_inf}"
                    )
                    self.assertEqual(
                        0,
                        num_nan,
                        f"Expected num_nan to be 1, but received {num_nan}, place={place}.",
                    )

                paddle.amp.debugging.disable_tensor_checker()
                _assert_flag(False)


class TestCheckLayerNumerics(unittest.TestCase):
    def test_layer_checker(self):
        class MyLayer(paddle.nn.Layer):
            def __init__(self, dtype):
                super().__init__()
                self._w = self.create_parameter([2, 3], dtype=dtype)
                self._b = self.create_parameter([2, 3], dtype=dtype)

            @paddle.amp.debugging.check_layer_numerics
            def forward(self, x):
                return x * self._w + self._b

        dtype = 'float32'
        x = paddle.rand([10, 2, 3], dtype=dtype)
        model = MyLayer(dtype)
        loss = model(x)
        adam = paddle.optimizer.Adam(parameters=model.parameters())
        loss.backward()
        adam.step()

    def test_error_no_element(self):
        class MyLayer(paddle.nn.Layer):
            def __init__(self, dtype):
                super().__init__()
                self._w = self.create_parameter([2, 3], dtype=dtype)

            @paddle.amp.debugging.check_layer_numerics
            def forward(self):
                return self._w

        with self.assertRaises(RuntimeError):
            dtype = 'float32'
            model = MyLayer(dtype)
            data = model()

    def test_error_type_error(self):
        class MyLayer(paddle.nn.Layer):
            def __init__(self, dtype):
                super().__init__()
                self._w = self.create_parameter([2, 3], dtype=dtype)

            @paddle.amp.debugging.check_layer_numerics
            def forward(self, x):
                return self._w * x

        x = 1
        with self.assertRaises(RuntimeError):
            dtype = 'float32'
            model = MyLayer(dtype)
            data = model(x)


if __name__ == '__main__':
    unittest.main()
