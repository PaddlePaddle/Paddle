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

import copy
import os
import subprocess
import sys
import unittest

import numpy as np

import paddle
from paddle.framework import in_pir_mode


class TestNanInfBase(unittest.TestCase):
    def setUp(self):
        self._python_interp = sys.executable
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            self._python_interp += " -m coverage run --branch -p"

        self.env = os.environ.copy()
        paddle.disable_static()

    def run_command(self, cmd):
        print(f"Run command: {cmd}")
        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )

        out, err = proc.communicate()
        returncode = proc.returncode
        return returncode, out, err

    def generate_inputs(self, shape, dtype="float32"):
        data = np.random.random(size=shape).astype(dtype)
        # [-10, 10)
        x = (data * 20 - 10) * np.random.randint(
            low=0, high=2, size=shape
        ).astype(dtype)
        y = np.random.randint(low=0, high=2, size=shape).astype(dtype)
        return x, y


class TestNanInf(TestNanInfBase):
    def setUp(self):
        super().setUp()
        self.check_static = True
        self.check_dygraph = True
        self.check_nan_inf_level = 0
        self.dygraph_expected_op_count = {"divide": 1}

    def check_op_count(self, log, expected_op_count=None):
        if expected_op_count is None:
            return

        lines = copy.copy(log).decode().split("\n")
        actual_op_count = {}
        tensor_info_list = paddle.amp.accuracy_compare.parse_lines(lines)
        for tensor_info in tensor_info_list:
            print(tensor_info)
            if actual_op_count.get(tensor_info.op_type, None) is None:
                actual_op_count[tensor_info.op_type] = 1
            else:
                actual_op_count[tensor_info.op_type] += 1
        print(actual_op_count)

        for op_type, expected_value in expected_op_count.items():
            actual_value = actual_op_count.get(op_type, 0)
            self.assertEqual(
                actual_value,
                expected_value,
                f"The number of operator < {op_type} > is expected to be {expected_value}, but received {actual_value}.",
            )
        print()

    def run_check_nan_inf(self, cmd, expected_op_count=None):
        returncode, out, err = self.run_command(cmd)
        self.check_op_count(out, expected_op_count)
        if self.check_nan_inf_level == 0:
            # in python3, type(out+err) is 'bytes', need use encode
            self.assertNotEqual(
                (out + err).find(b'There are NAN or INF'),
                -1,
                f"Cannot find NAN / INF keyword in:\n{out + err}",
            )

    def test_nan_inf_static(self):
        if not self.check_static:
            return

        filepath = os.path.dirname(__file__) + "/check_nan_inf_base.py"
        cmd = f"{self._python_interp} {filepath}"
        self.run_check_nan_inf(cmd, None)

    def test_nan_inf_dynamic(self):
        if not self.check_dygraph:
            return

        filepath = os.path.dirname(__file__) + "/check_nan_inf_base_dygraph.py"

        # Test on CPU.
        cmd = f"{self._python_interp} {filepath} --check_nan_inf_level {self.check_nan_inf_level}"
        self.run_check_nan_inf(cmd, self.dygraph_expected_op_count)

        # Test on GPU.
        if paddle.base.core.is_compiled_with_cuda():
            cmd = f"{self._python_interp} {filepath} --use_cuda --check_nan_inf_level {self.check_nan_inf_level}"
            self.run_check_nan_inf(cmd, self.dygraph_expected_op_count)


class TestCheckAll(TestNanInf):
    def setUp(self):
        super().setUp()
        self.check_static = False
        self.check_dygraph = True
        self.check_nan_inf_level = 3
        self.dygraph_expected_op_count = {
            'assign_value_': 2,
            'full_': 3,
            'matmul': 2,
            'add': 2,
            'sigmoid': 1,
            'cast': 1,
            'divide': 1,
            'softmax': 1,
            'mean': 1,
            'mean_grad': 1,
            'softmax_grad': 1,
            'divide_grad': 1,
            'add_grad': 4,
            'matmul_grad': 3,
            'sigmoid_grad': 1,
            'sgd_': 4,
        }


class TestNanInfEnv(TestNanInf):
    def setUp(self):
        super().setUp()
        # windows python have some bug with env, so need use str to pass ci
        # otherwise, "TypeError: environment can only contain strings"
        self.env["PADDLE_INF_NAN_SKIP_OP"] = "mul"
        self.env["PADDLE_INF_NAN_SKIP_ROLE"] = "loss"
        self.env["PADDLE_INF_NAN_SKIP_VAR"] = "elementwise_add:fc_0.tmp_1"

        self.check_static = True
        self.check_dygraph = False
        self.check_nan_inf_level = 0
        self.dygraph_expected_op_count = None


class TestNanInfStack(TestNanInfBase):
    def check_stack(self, file_name):
        cmd = self._python_interp + file_name
        returncode, out, err = self.run_command(cmd)

        print(out)
        print(err)

        # in python3, type(out+err) is 'bytes', need use encode
        assert (out + err).find(b' z = paddle.pow(x, y)') != -1

    def test_check_stack(self):
        self.check_stack(" check_nan_inf_backward_stack.py")

    def test_static_check_stack(self):
        if not paddle.framework.use_pir_api() and not os.environ.get(
            "FLAGS_enable_pir_api"
        ):
            self.check_stack(" check_nan_inf_backward_static_stack.py")


class TestNanInfCheckResult(TestNanInfBase):
    def get_reference_num_nan_inf(self, x):
        out = np.log(x)
        num_nan = np.sum(np.isnan(out))
        num_inf = np.sum(np.isinf(out))
        print(f"[reference] num_nan={num_nan}, num_inf={num_inf}")
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
                raise AssertionError
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
            print(f"[paddle] num_nan={num_nan}, num_inf={num_inf}")
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
        if paddle.base.core.is_compiled_with_cuda():
            _check_num_nan_inf(use_cuda=True)

    def run_check_nan_inf_level(self, use_cuda, dtype, level):
        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": level}
        )

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
        level = 2
        self.run_check_nan_inf_level(
            use_cuda=False, dtype="float32", level=level
        )
        if paddle.base.core.is_compiled_with_cuda():
            self.run_check_nan_inf_level(
                use_cuda=True, dtype="float32", level=level
            )

    def test_check_nan_inf_level_float16(self):
        level = 3
        self.run_check_nan_inf_level(
            use_cuda=False, dtype="float32", level=level
        )
        if paddle.base.core.is_compiled_with_cuda():
            self.run_check_nan_inf_level(
                use_cuda=True, dtype="float16", level=level
            )


class TestCheckNumericsAPI(TestNanInfBase):
    def test_eager(self):
        shape = [8, 8]
        x_np, y_np = self.generate_inputs(shape, "float32")

        device_list = ["cpu"]
        if paddle.base.core.is_compiled_with_cuda():
            device_list.append("gpu:0")

        for device in device_list:
            paddle.device.set_device(device)
            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            paddle.amp.debugging.check_numerics(
                tensor=x,
                op_type="to_tensor",
                var_name="x",
                debug_mode=paddle.amp.debugging.DebugMode.CHECK_ALL,
            )
            paddle.amp.debugging.check_numerics(
                tensor=y,
                op_type="to_tensor",
                var_name="y",
                debug_mode=paddle.amp.debugging.DebugMode.CHECK_ALL,
            )

    def test_static(self):
        paddle.enable_static()
        shape = [8, 8]
        x_np, y_np = self.generate_inputs(shape, "float32")

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name='x', shape=[8, 8], dtype="float32")
            y = paddle.static.data(name='y', shape=[8, 8], dtype="float32")
            out = paddle.add(x, y)
            if in_pir_mode():
                paddle.amp.debugging.check_numerics(
                    tensor=out,
                    op_type="elementwise_add",
                    var_name=out.id,
                    debug_mode=paddle.amp.debugging.DebugMode.CHECK_ALL,
                )
            else:
                paddle.amp.debugging.check_numerics(
                    tensor=out,
                    op_type="elementwise_add",
                    var_name=out.name,
                    debug_mode=paddle.amp.debugging.DebugMode.CHECK_ALL,
                )
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(main_program, feed={"x": x_np, "y": y_np}, fetch_list=[out])
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
