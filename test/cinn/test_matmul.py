#!/usr/bin/env python3

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

import unittest

import cinn
import numpy as np
from cinn import Target, ir, lang, runtime, utils
from cinn.poly import create_stages


class TestMamul(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k32
        self.target.os = Target.OS.Linux
        self.m = 1024
        self.n = 1024
        self.k = 1024
        self.bn = 32

        self.engine = cinn.ExecutionEngine()
        utils.ProfilerHelper.enable_cpu()
        self.assertTrue(utils.ProfilerHelper.is_enable_cpu())

    def test_matmul_basic(self):
        a, b, c, c_target, *args = create_data(self.m, self.n, self.k, self.bn)
        module = create_matmul_basic(self.target, self.m, self.n, self.k)

        self.engine.link(module)
        matmul = self.engine.lookup("matmul")
        matmul(args)
        cd = c.numpy()
        cd_target = c_target.numpy()
        np.testing.assert_allclose(cd, cd_target, atol=1e-4, rtol=1e-5)
        print(utils.HostEventRecorder.table())

    def test_matmul_tile(self):
        a, b, c, c_target, *args = create_data(self.m, self.n, self.k, self.bn)
        module = create_matmul_tile(self.target, self.m, self.n, self.k)
        print('module:\n', module.get_c_code())
        self.engine.link(module)
        matmul = self.engine.lookup("matmul_tile")
        matmul(args)
        cd = c.numpy()
        cd_target = c_target.numpy()
        np.testing.assert_allclose(cd, cd_target, atol=1e-4, rtol=1e-5)


def create_matmul_basic(target, m, n, k):
    m, n, k = (ir.Expr(_) for _ in (m, n, k))

    a = lang.Placeholder("float32", "A", [m, k])
    b = lang.Placeholder("float32", "B", [k, n])

    k1 = ir.Var(k.as_int32(), "k1")
    c = lang.compute(
        [m, n],
        lambda v: lang.reduce_sum(
            a(v[0], k1.to_expr_mutable()) * b(k1.to_expr_mutable(), v[1]), [k1]
        ),
        "c",
    )

    stages = create_stages([c])
    c_stage = stages[c]

    builder = lang.Module.Builder("matmul", target)

    ts = [a.to_tensor(), b.to_tensor(), c]
    func = lang.lower("matmul", stages, ts)
    print('func', func)
    builder.add_function(func)
    return builder.build()


def create_matmul_tile(target, m, n, k):
    m, n, k = (ir.Expr(_) for _ in [m, n, k])
    a = lang.Placeholder("float32", "A", [m, k])
    b = lang.Placeholder("float32", "B", [k, n])

    k1 = ir.Var(k.as_int32(), "k1")
    c = lang.compute(
        [m, n],
        lambda v: lang.reduce_sum(
            a(v[0], k1.to_expr_mutable()) * b(k1.to_expr_mutable(), v[1]), [k1]
        ),
        "c",
    )

    stages = create_stages([c])
    stages[c].tile(0, 1, 4, 4)

    builder = lang.Module.Builder("matmul_tile", target)
    ts = [a.to_tensor(), b.to_tensor(), c]
    func = lang.lower("matmul_tile", stages, ts)
    print('func', func)
    builder.add_function(func)
    return builder.build()


def create_data(m, n, k, bn):
    # call around to lower the numpy's float precision so that it will not vary too much from C's float precision.
    a_init = np.around(np.random.randn(m, k).astype("float32"), 2)
    b_init = np.around(np.random.randn(k, n).astype("float32"), 2)
    a = runtime.cinn_buffer_t(a_init, runtime.cinn_x86_device)
    b = runtime.cinn_buffer_t(b_init, runtime.cinn_x86_device)
    c = runtime.cinn_buffer_t(
        np.zeros([m, n]).astype("float32"), runtime.cinn_x86_device
    )
    c_target = runtime.cinn_buffer_t(
        a.numpy() @ b.numpy(), runtime.cinn_x86_device
    )
    packed_b = runtime.cinn_buffer_t(
        np.zeros([n // bn, k, bn]).astype("float32"), runtime.cinn_x86_device
    )

    a_arg = runtime.cinn_pod_value_t(a)
    b_arg = runtime.cinn_pod_value_t(b)
    c_arg = runtime.cinn_pod_value_t(c)
    packed_b_arg = runtime.cinn_pod_value_t(packed_b)
    return [a, b, c, c_target, a_arg, b_arg, c_arg]


if __name__ == "__main__":
    unittest.main()
