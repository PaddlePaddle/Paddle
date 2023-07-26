# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import logging
import os
import struct
import unittest

import numpy as np
from cinn.common import (
    BFloat16,
    Bool,
    DefaultHostTarget,
    DefaultNVGPUTarget,
    Float,
    Float16,
    Int,
    UInt,
    is_compiled_with_cuda,
)
from cinn.runtime import seed as cinn_seed

import paddle

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="op_test")


def convert_float_to_uint16(data, data_format="NCHW"):
    if data.size == 0:
        return data.view(np.uint16)

    if data_format == "NHWC":
        data = np.transpose(data, [0, 3, 1, 2])

    new_data = np.vectorize(
        lambda x: struct.unpack('<I', struct.pack('<f', x))[0] >> 16,
        otypes=[np.uint16],
    )(data.flat)
    new_data = np.reshape(new_data, data.shape)

    if data_format == "NHWC":
        new_data = np.transpose(new_data, [0, 2, 3, 1])
    return new_data


def convert_uint16_to_float(data):
    new_data = np.vectorize(
        lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0],
        otypes=[np.float32],
    )(data.flat)
    return np.reshape(new_data, data.shape)


class OpTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_target()
        self._init_results()
        self._init_seed()

    def _init_seed(self, seed_value=1234):
        np.random.seed(seed_value)
        paddle.seed(seed_value)
        cinn_seed(seed_value)

    def _init_results(self):
        self.paddle_outputs = []
        self.paddle_grads = []
        self.cinn_outputs = []
        self.cinn_grads = []

    def _init_target(self):
        self.target = DefaultHostTarget()
        if is_compiled_with_cuda():
            self.target = DefaultNVGPUTarget()

    def _get_device(self):
        return "NVGPU" if is_compiled_with_cuda() else "CPU"

    def build_paddle_program(self, target):
        raise Exception("Not implemented.")

    def get_paddle_grads(self, outputs, inputs, grad_outputs):
        grad_tensors = []
        for grad in grad_outputs:
            grad_tensors.append(paddle.to_tensor(grad))
        grads = paddle.grad(outputs, inputs, grad_tensors)

        return grads

    def build_cinn_program(self, target):
        raise Exception("Not implemented.")

    def get_cinn_output(
        self, prog, target, inputs, feed_data, outputs, passes=[], scope=None
    ):
        fetch_ids = {str(out) for out in outputs}
        result = prog.build_and_get_output(
            target, inputs, feed_data, outputs, passes=passes, scope=scope
        )
        outs_and_grads = []
        for res in result:
            outs_and_grads.append(res.numpy(target))

        return outs_and_grads

    def apply_pass(self, prog, target, passes=["Decomposer"], fetch_ids=set()):
        def print_program(prog):
            if logger.getEffectiveLevel() != logging.DEBUG:
                return
            for i in range(prog.size()):
                print(prog[i])

        logger.debug("============ Before Decomposer Pass ============")
        print_program(prog)

        prog.apply_pass(fetch_ids, target, passes)

        logger.debug("============ After Decomposer Pass ============")
        print_program(prog)

    def check_outputs_and_grads(
        self,
        max_relative_error=1e-5,
        max_absolute_error=1e-6,
        all_equal=False,
        equal_nan=False,
    ):
        self.build_paddle_program(self.target)
        self.build_cinn_program(self.target)

        logger.debug("============ Check Outputs ============")
        self.check_results(
            self.paddle_outputs,
            self.cinn_outputs,
            max_relative_error,
            max_absolute_error,
            all_equal,
            equal_nan,
            "Outputs",
        )

        if len(self.cinn_grads) != 0:
            logger.debug("============ Check Grads ============")
            self.check_results(
                self.paddle_grads,
                self.cinn_grads,
                max_relative_error,
                max_absolute_error,
                all_equal,
                equal_nan,
                "Grads",
            )

    def check_results(
        self,
        expect_res,
        actual_res,
        max_relative_error,
        max_absolute_error,
        all_equal=False,
        equal_nan=False,
        name="Outputs",
    ):
        def _compute_error_message(output_id, expect, actual):
            absolute_diff = np.abs(expect - actual).flatten()
            relative_diff = absolute_diff / np.abs(expect).flatten()
            max_relative_diff = 0
            max_absolute_diff = 0
            offset = -1
            num_diffs = 0
            for i in range(len(relative_diff)):
                if relative_diff[i] > max_relative_diff:
                    max_relative_diff = relative_diff[i]
                if absolute_diff[i] > max_absolute_diff:
                    max_absolute_diff = absolute_diff[i]
                if (
                    relative_diff[i] > max_relative_error
                    or absolute_diff[i] > max_absolute_error
                ):
                    num_diffs = num_diffs + 1
                    offset = i if offset == -1 else offset
                    # The following print can be used to debug.
                    # print("i=%d, %e vs %e, relative_diff=%e, absolute_diff=%e" % (i, expect.flatten()[i], actual.flatten()[i], relative_diff[i], absolute_diff[i]))
            error_message = (
                "[%s] The %d-th output: total %d different results, offset=%d, shape=%s, %e vs %e. Maximum diff of the whole array: maximum_relative_diff=%e, maximum_absolute_diff=%e."
                % (
                    self._get_device(),
                    output_id,
                    num_diffs,
                    offset,
                    str(expect.shape),
                    expect.flatten()[offset],
                    actual.flatten()[offset],
                    max_relative_diff,
                    max_absolute_diff,
                )
            )
            return error_message

        def _check_error_message(output_id, expect, actual):
            expect_flatten = expect.flatten()
            actual_flatten = actual.flatten()
            self.assertEqual(
                len(expect_flatten),
                len(actual_flatten),
                "[{}] The {}-th output size different, which expect shape is {} but actual is {}.".format(
                    self._get_device(), output_id, expect.shape, actual.shape
                ),
            )
            num_diffs = 0
            offset = -1
            for i in range(len(expect_flatten)):
                if expect_flatten[i] != actual_flatten[i]:
                    num_diffs = num_diffs + 1
                    offset = i if offset == -1 else offset

            error_message = "[{}] The {}-th output: total {} different results, the first different result's offset={}, where expect value is {} but actual is {}.".format(
                self._get_device(),
                output_id,
                num_diffs,
                offset,
                expect_flatten[offset],
                actual_flatten[offset],
            )
            return error_message

        self.assertEqual(len(expect_res), len(actual_res))
        for i in range(len(expect_res)):
            if expect_res[i] is None:
                continue

            if isinstance(expect_res[i], paddle.Tensor):
                expect = expect_res[i].numpy()
            else:
                expect = expect_res[i]
            actual = actual_res[i]

            # data conversion for bfloat16 (uint16 in numpy)
            if actual.dtype == np.uint16:
                max_relative_error = 1e-2
                if expect.dtype == np.float32 or expect.dtype == np.float64:
                    actual = convert_uint16_to_float(actual)

            self.assertEqual(
                expect.dtype,
                actual.dtype,
                msg="[{}] The {}-th output dtype different, which expect shape is {} but actual is {}.".format(
                    self._get_device(), i, expect.dtype, actual.dtype
                ),
            )
            # NOTE: Paddle's 0D Tensor will be changed to 1D when calling tensor.numpy(),
            # only check non-0D Tensor's shape here. 0D-Tensor's shape will be verified by `test_zero_dim_tensor.py`
            if len(expect.shape) != 0 and len(actual.shape) != 0:
                self.assertEqual(
                    expect.shape,
                    actual.shape,
                    msg="[{}] The {}-th output shape different, which expect shape is {} but actual is {}.".format(
                        self._get_device(), i, expect.shape, actual.shape
                    ),
                )

            should_all_equal = all_equal or (
                actual.dtype
                in [np.dtype('bool'), np.dtype('int32'), np.dtype('int64')]
            )

            if expect.dtype == np.uint16:
                expect_float = convert_uint16_to_float(expect)
            if actual.dtype == np.uint16:
                actual_float = convert_uint16_to_float(actual)

            is_allclose = True
            error_message = ""
            if not should_all_equal:
                is_allclose = np.allclose(
                    expect,
                    actual,
                    atol=max_absolute_error,
                    rtol=max_relative_error,
                    equal_nan=equal_nan,
                )
                # _compute_error_message checks which values have absolute or relative error
                error_message = (
                    "np.allclose(expect, actual, atol={}, rtol={}) checks succeed!".format(
                        max_absolute_error, max_relative_error
                    )
                    if is_allclose
                    else _compute_error_message(i, expect, actual)
                )
            else:
                is_allclose = np.all(expect == actual)
                # _check_error_message checks which values are not equal
                error_message = (
                    "(expect == actual) checks succeed!"
                    if is_allclose
                    else _check_error_message(i, expect, actual)
                )

            error_message = "[Check " + name + "] " + error_message

            logger.debug(f"{is_allclose} {error_message}")
            self.assertTrue(is_allclose, msg=error_message)

    @staticmethod
    def nptype2cinntype(dtype):
        switch_map = {
            # numpy has no 'bfloat16', we use uint16 to hold bfloat16 data, same to Paddle
            "uint16": BFloat16(),
            "bfloat16": BFloat16(),
            "float16": Float16(),
            "float32": Float(32),
            "float64": Float(64),
            "int8": Int(8),
            "int16": Int(16),
            "int32": Int(32),
            "int64": Int(64),
            "uint8": UInt(8),
            # numpy has no 'bfloat16', we use uint16 to hold bfloat16 data, same to Paddle
            # "uint16": UInt(16),
            "uint32": UInt(32),
            "uint64": UInt(64),
            "bool": Bool(),
        }
        assert str(dtype) in switch_map, str(dtype) + " not support in CINN"
        return switch_map[str(dtype)]

    @staticmethod
    def paddleddtype2cinntype(dtype):
        return OpTest.nptype2cinntype(OpTest.paddleddtype2str(dtype))

    @staticmethod
    def random(shape, dtype="float32", low=0.0, high=1.0):
        assert bool(shape), "Shape should not empty!"
        assert -1 not in shape, "Shape should not -1!"
        if dtype in ["float16", "float32", "float64"]:
            return np.random.uniform(low, high, shape).astype(dtype)
        elif dtype == "bfloat16":
            return convert_float_to_uint16(
                np.random.uniform(low, high, shape).astype("float32")
            )
        elif dtype == "bool":
            return np.random.choice(a=[False, True], size=shape).astype(dtype)
        elif dtype in [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
        ]:
            return np.random.randint(low, high, shape).astype(dtype)
        else:
            raise Exception("Not supported yet.")


class OpTestTool:
    @classmethod
    def skip_if(cls, condition: object, reason: str):
        return unittest.skipIf(condition, reason)
