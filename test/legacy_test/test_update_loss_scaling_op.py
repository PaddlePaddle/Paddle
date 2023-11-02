#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.static.amp import amp_nn


def update_loss_scaling_wrapper(
    x,
    found_inf,
    prev_loss_scaling,
    num_good_steps,
    num_bad_steps,
    incr_every_n_steps,
    decr_every_n_nan_or_inf,
    incr_ratio,
    decr_ratio,
    stop_update=False,
):
    amp_nn.update_loss_scaling(
        [x],
        found_inf,
        prev_loss_scaling,
        num_good_steps,
        num_bad_steps,
        incr_every_n_steps,
        decr_every_n_nan_or_inf,
        incr_ratio,
        decr_ratio,
        stop_update,
    )
    return x, prev_loss_scaling, num_good_steps, num_bad_steps


class TestUpdateLossScalingOp(OpTest):
    def setUp(self):
        self.op_type = "update_loss_scaling"
        self.init()
        self.python_api = update_loss_scaling_wrapper
        self.python_out_sig = [
            "out0",
            "LossScaling",
            "OutGoodSteps",
            "OutBadSteps",
        ]
        found_inf = np.array([False], dtype=np.bool_)
        x = np.random.random((1024, 1024)).astype(self.dtype)

        self.inputs = {
            'X': [('x0', x)],
            'FoundInfinite': found_inf,
            'PrevLossScaling': self.prev_loss_scaling,
            'InGoodSteps': self.num_good_steps,
            'InBadSteps': self.num_bad_steps,
        }

        self.outputs = {
            'Out': [('out0', x)],
            'LossScaling': self.prev_loss_scaling * self.incr_ratio,
            'OutGoodSteps': self.zero_steps,
            'OutBadSteps': self.zero_steps,
        }

    def init(self):
        self.incr_ratio = 2.0
        self.decr_ratio = 0.8
        self.init_dtype()
        self.prev_loss_scaling = np.array([2048]).astype(
            self.loss_scaling_dtype
        )
        self.num_good_steps = np.array([999], dtype=np.int32)
        self.num_bad_steps = np.array([1], dtype=np.int32)
        self.zero_steps = np.array([0], dtype=np.int32)
        self.stop_update = np.array([False], dtype=np.bool_)
        self.attrs = {
            'incr_every_n_steps': 1000,
            'decr_every_n_nan_or_inf': 2,
            'incr_ratio': self.incr_ratio,
            'decr_ratio': self.decr_ratio,
        }

    def init_dtype(self):
        self.dtype = np.float32
        self.loss_scaling_dtype = np.float32

    def test_check_output(self):
        self.check_output(no_check_set=['Out'])


class TestUpdateLossScalingFP16Op(TestUpdateLossScalingOp):
    def init_dtype(self):
        self.dtype = np.float16
        self.loss_scaling_dtype = np.float32


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestUpdateLossScalingBF16Op(OpTest):
    def init(self):
        self.incr_ratio = 2.0
        self.decr_ratio = 0.8
        self.dtype = np.uint16
        self.np_dtype = "float32"
        self.prev_loss_scaling = np.array([2048]).astype(self.np_dtype)
        self.num_good_steps = np.array([999], dtype=np.int32)
        self.num_bad_steps = np.array([1], dtype=np.int32)
        self.zero_steps = np.array([0], dtype=np.int32)
        self.stop_update = np.array([False], dtype=np.bool_)
        self.attrs = {
            'incr_every_n_steps': 1000,
            'decr_every_n_nan_or_inf': 2,
            'incr_ratio': self.incr_ratio,
            'decr_ratio': self.decr_ratio,
        }

    def setUp(self):
        self.op_type = "update_loss_scaling"
        self.init()
        self.python_api = update_loss_scaling_wrapper
        self.python_out_sig = [
            "out0",
            "LossScaling",
            "OutGoodSteps",
            "OutBadSteps",
        ]
        found_inf = np.array([False], dtype=np.bool_)
        x = np.random.random((1024, 1024)).astype(self.np_dtype)

        self.inputs = {
            'X': [('x0', convert_float_to_uint16(x))],
            'FoundInfinite': found_inf,
            # do not convert
            'PrevLossScaling': self.prev_loss_scaling,
            'InGoodSteps': self.num_good_steps,
            'InBadSteps': self.num_bad_steps,
        }

        self.outputs = {
            'Out': [('out0', convert_float_to_uint16(x))],
            # do not convert
            'LossScaling': self.prev_loss_scaling * self.incr_ratio,
            'OutGoodSteps': self.zero_steps,
            'OutBadSteps': self.zero_steps,
        }

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), no_check_set=['Out'])


class TestUpdateLossScalingOpBad(TestUpdateLossScalingOp):
    def setUp(self):
        self.op_type = "update_loss_scaling"
        self.init()
        self.python_api = update_loss_scaling_wrapper
        self.python_out_sig = [
            "out0",
            "LossScaling",
            "OutGoodSteps",
            "OutBadSteps",
        ]
        found_inf = np.array([True], dtype=np.bool_)
        x = np.random.random((1024, 1024)).astype(self.dtype)
        i = np.random.randint(0, 1024, 1)
        j = np.random.randint(0, 1024, 1)
        x[i[0]][j[0]] = np.inf

        self.inputs = {
            'X': [('x0', x)],
            'FoundInfinite': found_inf,
            'PrevLossScaling': self.prev_loss_scaling,
            'InGoodSteps': self.num_good_steps,
            'InBadSteps': self.num_bad_steps,
            'StopUpdate': self.stop_update,
        }

        self.outputs = {
            'Out': [('out0', np.zeros_like(x))],
            'LossScaling': self.prev_loss_scaling * self.decr_ratio,
            'OutGoodSteps': self.zero_steps,
            'OutBadSteps': self.zero_steps,
        }

    def test_check_output(self):
        self.check_output()


class TestUpdateLossScalingLayer(unittest.TestCase):
    def loss_scaling_check(self, use_cuda=True, scope=base.Scope()):
        with paddle_static_guard():
            a = paddle.static.data(
                name="a", shape=[1024, 1024], dtype='float32'
            )
            b = paddle.static.data(name="b", shape=[512, 128], dtype='float32')
            x = [a, b]
            found_inf = paddle.static.data(
                name="found_inf", shape=[1], dtype='bool'
            )
            prev_loss_scaling = paddle.static.data(
                name="prev_loss_scaling", shape=[1], dtype='float32'
            )
            num_good_steps = paddle.static.data(
                name="num_good_steps", shape=[1], dtype='int32'
            )
            num_bad_steps = paddle.static.data(
                name="num_bad_steps", shape=[1], dtype='int32'
            )

            a_v = np.random.random([1024, 1024]).astype('float32')
            b_v = np.random.random([512, 128]).astype('float32')
            found_inf_v = np.array([False]).astype('bool')
            prev_loss_scaling_v = np.array([2048]).astype('float32')
            num_good_steps_v = np.array([999], dtype=np.int32)
            num_bad_steps_v = np.array([1], dtype=np.int32)

            incr_every_n_steps = 1000
            decr_every_n_nan_or_inf = 2
            incr_ratio = 2
            decr_ratio = 0.8

            result = amp_nn.update_loss_scaling(
                x,
                found_inf,
                prev_loss_scaling,
                num_good_steps,
                num_bad_steps,
                incr_every_n_steps,
                decr_every_n_nan_or_inf,
                incr_ratio,
                decr_ratio,
                name="update_loss_scaling",
            )

            place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
            exe = base.Executor(place)
            with base.scope_guard(scope):
                exe.run(base.default_startup_program())
                result_v = exe.run(
                    feed={
                        'a': a_v,
                        'b': b_v,
                        'found_inf': found_inf_v,
                        'prev_loss_scaling': prev_loss_scaling_v,
                        'num_good_steps': num_good_steps_v,
                        'num_bad_steps': num_bad_steps_v,
                    },
                    fetch_list=[
                        result,
                        x,
                        found_inf,
                        prev_loss_scaling,
                        num_good_steps,
                        num_bad_steps,
                    ],
                )

            np.testing.assert_array_equal(result_v[0], a_v)
            np.testing.assert_array_equal(result_v[1], b_v)
            np.testing.assert_array_equal(result_v[0], result_v[2])
            np.testing.assert_array_equal(result_v[1], result_v[3])
            np.testing.assert_array_equal(result_v[4], found_inf_v)
            np.testing.assert_array_equal(
                result_v[5], prev_loss_scaling_v * incr_ratio
            )
            np.testing.assert_array_equal(
                result_v[6], np.zeros_like(num_good_steps_v)
            )
            np.testing.assert_array_equal(
                result_v[7], np.zeros_like(num_bad_steps_v)
            )

    def loss_scaling_check_inf(self, use_cuda=True, scope=base.Scope()):
        with paddle_static_guard():
            a = paddle.static.data(
                name="a", shape=[1024, 1024], dtype='float32'
            )
            b = paddle.static.data(name="b", shape=[512, 128], dtype='float32')
            x = [a, b]
            found_inf = paddle.static.data(
                name="found_inf", shape=[1], dtype='bool'
            )
            prev_loss_scaling = paddle.static.data(
                name="prev_loss_scaling", shape=[1], dtype='float32'
            )
            num_good_steps = paddle.static.data(
                name="num_good_steps", shape=[1], dtype='int32'
            )
            num_bad_steps = paddle.static.data(
                name="num_bad_steps", shape=[1], dtype='int32'
            )

            a_v = np.random.random([1024, 1024]).astype('float32')
            b_v = np.random.random([512, 128]).astype('float32')
            i = np.random.randint(0, 1024, 1)
            j = np.random.randint(0, 1024, 1)
            a_v[i[0]][j[0]] = np.inf
            found_inf_v = np.array([True]).astype('bool')
            prev_loss_scaling_v = np.array([2048]).astype('float32')
            num_good_steps_v = np.array([999], dtype=np.int32)
            num_bad_steps_v = np.array([1], dtype=np.int32)

            incr_every_n_steps = 1000
            decr_every_n_nan_or_inf = 2
            incr_ratio = 2
            decr_ratio = 0.8

            result = amp_nn.update_loss_scaling(
                x,
                found_inf,
                prev_loss_scaling,
                num_good_steps,
                num_bad_steps,
                incr_every_n_steps,
                decr_every_n_nan_or_inf,
                incr_ratio,
                decr_ratio,
                name="update_loss_scaling",
            )

            place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
            exe = base.Executor(place)
            with base.scope_guard(scope):
                exe.run(base.default_startup_program())
                result_v = exe.run(
                    feed={
                        'a': a_v,
                        'b': b_v,
                        'found_inf': found_inf_v,
                        'prev_loss_scaling': prev_loss_scaling_v,
                        'num_good_steps': num_good_steps_v,
                        'num_bad_steps': num_bad_steps_v,
                    },
                    fetch_list=[
                        result,
                        x,
                        found_inf,
                        prev_loss_scaling,
                        num_good_steps,
                        num_bad_steps,
                    ],
                )
            np.testing.assert_array_equal(result_v[0], np.zeros_like(a_v))
            np.testing.assert_array_equal(result_v[1], np.zeros_like(b_v))
            np.testing.assert_array_equal(result_v[2], np.zeros_like(a_v))
            np.testing.assert_array_equal(result_v[3], np.zeros_like(b_v))
            np.testing.assert_array_equal(result_v[4], found_inf_v)
            np.testing.assert_array_equal(
                result_v[5], prev_loss_scaling_v * decr_ratio
            )
            np.testing.assert_array_equal(
                result_v[6], np.zeros_like(num_good_steps_v)
            )
            np.testing.assert_array_equal(
                result_v[7], np.zeros_like(num_bad_steps_v)
            )

    def test_loss_scaling_cpu(self):
        with paddle_static_guard():
            main = base.Program()
            startup = base.Program()
            with base.unique_name.guard():
                with base.program_guard(main, startup):
                    self.loss_scaling_check(use_cuda=False)

    def test_loss_scaling_cpu_inf(self):
        with paddle_static_guard():
            main = base.Program()
            startup = base.Program()
            with base.unique_name.guard():
                with base.program_guard(main, startup):
                    self.loss_scaling_check_inf(use_cuda=False)

    def test_loss_scaling_gpu(self):
        if base.core.is_compiled_with_cuda():
            with paddle_static_guard():
                main = base.Program()
                startup = base.Program()
                with base.unique_name.guard():
                    with base.program_guard(main, startup):
                        self.loss_scaling_check(use_cuda=True)

    def test_loss_scaling_gpu_inf(self):
        if base.core.is_compiled_with_cuda():
            with paddle_static_guard():
                main = base.Program()
                startup = base.Program()
                with base.unique_name.guard():
                    with base.program_guard(main, startup):
                        self.loss_scaling_check_inf(use_cuda=True)


if __name__ == '__main__':
    unittest.main()
