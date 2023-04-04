# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest

import numpy as np

import paddle


class TestNanInfDirCheckResult(unittest.TestCase):
    def generate_inputs(self, shape, dtype="float32"):
        data = np.random.random(size=shape).astype(dtype)
        # [-10, 10)
        x = (data * 20 - 10) * np.random.randint(
            low=0, high=2, size=shape
        ).astype(dtype)
        y = np.random.randint(low=0, high=2, size=shape).astype(dtype)
        return x, y

    def get_reference_num_nan_inf(self, x):
        out = np.log(x)
        num_nan = np.sum(np.isnan(out))
        num_inf = np.sum(np.isinf(out))
        print(f"[reference] num_nan={num_nan}, num_inf={num_inf}")
        return num_nan, num_inf

    def get_num_nan_inf(
        self, x_np, use_cuda=True, add_assert=False, pt="nan_inf_log_dir"
    ):
        num_nan = 0
        num_inf = 0
        if add_assert:
            if use_cuda:
                paddle.device.set_device("gpu:0")
            else:
                paddle.device.set_device("cpu")
            x = paddle.to_tensor(x_np)
            out = paddle.log(x)
            sys.stdout.flush()
            if not use_cuda:
                os.path.exists(pt)
                num_nan = 0
                num_inf = 0
                for root, dirs, files in os.walk(pt):
                    for file_name in files:
                        if file_name.startswith('worker_cpu'):
                            file_path = os.path.join(root, file_name)
                            with open(file_path, "rb") as fp:
                                for e in fp:
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
                print(f"[paddle] num_nan={num_nan}, num_inf={num_inf}")
        return num_nan, num_inf

    def test_num_nan_inf(self):
        path = "nan_inf_log_dir"
        paddle.fluid.core.set_nan_inf_debug_path(path)

        def _check_num_nan_inf(use_cuda):
            shape = [32, 32]
            x_np, _ = self.generate_inputs(shape)
            num_nan_np, num_inf_np = self.get_reference_num_nan_inf(x_np)
            add_assert = (num_nan_np + num_inf_np) > 0
            num_nan, num_inf = self.get_num_nan_inf(
                x_np, use_cuda, add_assert, path
            )
            if not use_cuda:
                assert num_nan == num_nan_np and num_inf == num_inf_np

        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 3}
        )
        _check_num_nan_inf(use_cuda=False)
        if paddle.fluid.core.is_compiled_with_cuda():
            _check_num_nan_inf(use_cuda=True)
        x = paddle.to_tensor([2, 3, 4], 'float32')
        y = paddle.to_tensor([1, 5, 2], 'float32')
        z = paddle.add(x, y)


if __name__ == '__main__':
    unittest.main()
