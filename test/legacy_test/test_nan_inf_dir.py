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
import tempfile
import unittest

import numpy as np

import paddle


class TestNanInfDirCheckResult(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def generate_inputs(self, shape, low=0, high=1, dtype="float32"):
        data = np.random.random(size=shape).astype(dtype)
        x = (data * (high - low) + low) * np.random.randint(
            low=0, high=2, size=shape
        ).astype(dtype)
        return x

    def get_reference_num_nan_inf(self, x):
        out = np.log(x)
        num_nan = np.sum(np.isnan(out))
        num_inf = np.sum(np.isinf(out))
        print(f"-- [reference] num_nan={num_nan}, num_inf={num_inf}")
        return num_nan, num_inf

    def get_num_nan_inf(self, x_np, use_cuda=True, output_dir=None):
        if use_cuda:
            paddle.device.set_device("gpu:0")
        else:
            paddle.device.set_device("cpu")
        x = paddle.to_tensor(x_np)
        x = x * 0.5
        out = paddle.log(x)
        if use_cuda:
            paddle.device.cuda.synchronize()

        self.assertEqual(
            os.path.exists(output_dir) and os.path.isdir(output_dir), True
        )

        num_nan = 0
        num_inf = 0
        prefix = "worker_gpu" if use_cuda else "worker_cpu"
        for filename in os.listdir(output_dir):
            if filename.startswith(prefix):
                filepath = os.path.join(output_dir, filename)
                print(f"-- Parse {filepath}")
                with open(filepath, "rb") as fp:
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
        print(
            f"-- [paddle] use_cuda={use_cuda}, num_nan={num_nan}, num_inf={num_inf}"
        )
        return num_nan, num_inf

    def check_num_nan_inf(self, x_np, use_cuda, subdir):
        output_dir = self.temp_dir.name + "/" + subdir
        print(f"-- output_dir: {output_dir}")
        checker_config = paddle.amp.debugging.TensorCheckerConfig(
            enable=True,
            debug_mode=paddle.amp.debugging.DebugMode.CHECK_ALL,
            output_dir=output_dir,
        )
        paddle.amp.debugging.enable_tensor_checker(checker_config)

        num_nan_np, num_inf_np = self.get_reference_num_nan_inf(x_np)
        num_nan, num_inf = self.get_num_nan_inf(
            x_np,
            use_cuda,
            output_dir,
        )
        self.assertEqual(num_nan, num_nan_np)
        self.assertEqual(num_inf, num_inf_np)

        paddle.amp.debugging.disable_tensor_checker()

    def test_num_nan_inf(self):
        shape = [32, 32]
        x_np = self.generate_inputs(shape, -10, 10)
        self.check_num_nan_inf(
            x_np, use_cuda=False, subdir="check_nan_inf_dir_cpu"
        )
        if paddle.base.core.is_compiled_with_cuda():
            self.check_num_nan_inf(
                x_np, use_cuda=True, subdir="check_nan_inf_dir_gpu"
            )


if __name__ == '__main__':
    unittest.main()
