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
import subprocess
import sys
import unittest

import numpy as np

import paddle


class TestNanInf(unittest.TestCase):
    def setUp(self):
        self._python_interp = sys.executable
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            self._python_interp += " -m coverage run --branch -p"

        self.env = os.environ.copy()

    def check_nan_inf(self):
        cmd = self._python_interp

        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )

        out, err = proc.communicate()
        returncode = proc.returncode

        print(out)
        print(err)

        # in python3, type(out+err) is 'bytes', need use encode
        assert (out + err).find('There are NAN or INF'.encode()) != -1

    def test_nan_inf_in_static_mode(self):
        self._python_interp += " check_nan_inf_base.py"
        self.check_nan_inf()

    def test_nan_inf_in_dynamic_mode(self):
        self._python_interp += " check_nan_inf_base_dygraph.py"
        self.check_nan_inf()


class TestNanInfEnv(TestNanInf):
    def setUp(self):
        super().setUp()
        # windows python have some bug with env, so need use str to pass ci
        # otherwise, "TypeError: environment can only contain strings"
        self.env[str("PADDLE_INF_NAN_SKIP_OP")] = str("mul")
        self.env[str("PADDLE_INF_NAN_SKIP_ROLE")] = str("loss")
        self.env[str("PADDLE_INF_NAN_SKIP_VAR")] = str(
            "elementwise_add:fc_0.tmp_1"
        )


class TestNanInfCheckResult(unittest.TestCase):
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
        print("[reference] num_nan={}, num_inf={}".format(num_nan, num_inf))
        return num_nan, num_inf

    def get_num_nan_inf(self, x_np, use_cuda=True, add_assert=False):
        num_nan = 0
        num_inf = 0
        try:
            if use_cuda:
                paddle.device.set_device("gpu:0")
            else:
                paddle.device.set_device("cpu")
            x = paddle.to_tensor(x_np)
            out = paddle.log(x)
            sys.stdout.flush()
            if add_assert:
                assert False
        except Exception as e:
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
            print("[paddle] num_nan={}, num_inf={}".format(num_nan, num_inf))
        return num_nan, num_inf

    def test_num_nan_inf(self):
        def _check_num_nan_inf(use_cuda):
            shape = [32, 32]
            x_np, _ = self.generate_inputs(shape)
            num_nan_np, num_inf_np = self.get_reference_num_nan_inf(x_np)
            add_assert = (num_nan_np + num_inf_np) > 0
            num_nan, num_inf = self.get_num_nan_inf(x_np, use_cuda, add_assert)
            if not use_cuda:
                assert num_nan == num_nan_np and num_inf == num_inf_np

        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 0}
        )
        _check_num_nan_inf(use_cuda=False)
        if paddle.fluid.core.is_compiled_with_cuda():
            _check_num_nan_inf(use_cuda=True)

    def check_nan_inf_level(self, use_cuda, dtype):
        shape = [8, 8]
        x_np, y_np = self.generate_inputs(shape, dtype)

        if use_cuda:
            paddle.device.set_device("gpu:0")
        else:
            paddle.device.set_device("cpu")
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.log(x * 1e6) / y

    def test_check_nan_inf_level_float32(self):
        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 2}
        )
        self.check_nan_inf_level(use_cuda=False, dtype="float32")
        if paddle.fluid.core.is_compiled_with_cuda():
            self.check_nan_inf_level(use_cuda=True, dtype="float32")

    def test_check_nan_inf_level_float16(self):
        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 3}
        )
        if paddle.fluid.core.is_compiled_with_cuda():
            self.check_nan_inf_level(use_cuda=True, dtype="float16")


if __name__ == '__main__':
    unittest.main()
