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

import logging
import unittest

import numpy as np

import paddle
from paddle import fluid
from paddle.fluid import program_guard
from paddle.static.amp.amp_nn import check_finite_and_unscale

paddle.enable_static()


def compare(
    cpu_res, gcu_res, threshold=1.0e-5, rel_tol=1.0e-6, equal_nan=False
):
    assert len(cpu_res) == len(gcu_res)
    for i in range(len(cpu_res)):
        out = gcu_res[i]
        exp = cpu_res[i]
        assert out.shape == exp.shape
        assert out.dtype == exp.dtype
        if exp.dtype in [np.float16, np.float32, np.float64]:
            np.testing.assert_allclose(
                out, exp, rtol=rel_tol, atol=threshold, equal_nan=equal_nan
            )
        elif exp.dtype in [
            bool,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]:
            assert np.all(out == exp)
        else:
            assert logging.info('unsupport data type')
            assert 0


class TestCheckFiniteAndUnscale(unittest.TestCase):
    def get_prog(self, N):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with program_guard(main_program):
            A = []
            for i in range(N):
                a = paddle.static.data(
                    name="c%d" % i, shape=[32, 32], dtype='float32'
                )
                A.append(a)
            scale = paddle.static.data(name="scale", shape=[1], dtype='float32')

            out, found_inf = check_finite_and_unscale(A, scale)

        return main_program, out, found_inf

    def run_prog(self, c, scale, nonan=True):
        main_program, out, found_inf = self.get_prog(len(c))
        exe = fluid.Executor('gcu')
        vs = {}
        for i in range(len(c)):
            vs["c%d" % i] = c[i]
        vs['scale'] = scale
        out_, founf_inf_ = exe.run(
            main_program, feed=vs, fetch_list=[out, found_inf]
        )

        exe_cpu = fluid.Executor(fluid.CPUPlace())
        out_cpu, founf_inf_cpu = exe_cpu.run(
            main_program, feed=vs, fetch_list=[out, found_inf]
        )

        if not nonan:
            compare([out_, founf_inf_], [out_cpu, founf_inf_cpu])
        else:
            compare([founf_inf_], [founf_inf_cpu])

    def test_contains_nan(self):
        a = np.zeros((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')
        c = a / b
        scale = np.array([2.0]).astype('float32')

        self.run_prog([c], scale, True)

    def test_contains_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')
        scale = np.array([2.0]).astype('float32')
        c = a / b
        self.run_prog([c], scale, True)

    def test_not_contains_nan_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.ones((32, 32)).astype('float32')
        scale = np.array([2.0]).astype('float32')
        c = a / b
        self.run_prog([c], scale, False)


if __name__ == '__main__':
    unittest.main()
