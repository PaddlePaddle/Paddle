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
from cinn import Target, ir, lang, pe, runtime
from cinn.poly import create_stages


class TestPEReduction(unittest.TestCase):
    def setUp(self):
        self.m = 32
        self.n = 32

        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

        self.reduction_data = []

    def test_reduction_0(self):
        for fn_name, pe_fn, np_fn in [
            ("sum", pe.reduce_sum, np.sum),
            ("prod", pe.reduce_prod, np.prod),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [], True)

    def test_reduction_1(self):
        for fn_name, pe_fn, np_fn in [
            ("sum", pe.reduce_sum, np.sum),
            ("prod", pe.reduce_prod, np.prod),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [], False)

    def test_reduction_2(self):
        for fn_name, pe_fn, np_fn in [
            ("sum", pe.reduce_sum, np.sum),
            ("prod", pe.reduce_prod, np.prod),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [0], False)

    def test_reduction_3(self):
        for fn_name, pe_fn, np_fn in [
            ("sum", pe.reduce_sum, np.sum),
            ("prod", pe.reduce_prod, np.prod),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [0], True)

    def test_reduction_4(self):
        for fn_name, pe_fn, np_fn in [
            ("sum", pe.reduce_sum, np.sum),
            ("prod", pe.reduce_prod, np.prod),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [1], False)

    def test_reduction_5(self):
        for fn_name, pe_fn, np_fn in [
            ("sum", pe.reduce_sum, np.sum),
            ("prod", pe.reduce_prod, np.prod),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [1], True)

    def test_reduction_6(self):
        for fn_name, pe_fn, np_fn in [
            ("max", pe.reduce_max, np.max),
            ("min", pe.reduce_min, np.min),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [1], True)

    def test_reduction_7(self):
        for fn_name, pe_fn, np_fn in [
            ("max", pe.reduce_max, np.max),
            ("min", pe.reduce_min, np.min),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.reduction_tester(fn_name, pe_fn, np_fn, [1], False)

    def reduction_tester(self, fn_name, cinn_fn, np_fn, axes, keep_dims):
        m, n = (
            ir.Expr(_)
            for _ in (
                self.m,
                self.n,
            )
        )
        x = lang.Placeholder("float32", "x", [m, n])
        func_name = "test_" + fn_name
        y = cinn_fn(x.to_tensor(), axes, keep_dims)
        stages = create_stages([x.to_tensor(), y])
        func = lang.lower(func_name, stages, [x.to_tensor(), y])

        builder = lang.Module.Builder("reduction_module", self.target)
        builder.add_function(func)
        print(func)

        module = builder.build()
        self.compiler.build(module)

        fn = self.compiler.lookup(func_name)

        x_data, x_buf, out_buf, *args = self.create_data(axes, keep_dims)
        fn(args)

        np.testing.assert_allclose(
            out_buf.numpy(),
            self.create_target_data(x_data, np_fn, axes, keep_dims),
            atol=1e-4,
        )

    def create_target_data(self, x_data, np_target_fn, axes, keep_dims):
        axes_tuple = tuple(axes)
        if len(axes) == 0:
            axes_tuple = None
        return np_target_fn(x_data, axis=axes_tuple, keepdims=keep_dims)

    def create_data(self, axes, keep_dims):
        if not self.reduction_data:
            x_data = np.around(
                np.random.randn(self.m, self.n).astype("float32"), 2
            )
            x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
            if keep_dims:
                output_shape = [self.m, self.n]
                if axes:
                    for i in axes:
                        if i < 0:
                            i = i + len(output_shape)
                        output_shape[i] = 1
                else:
                    for i in range(len(output_shape)):
                        output_shape[i] = 1
            else:
                output_shape = [self.m, self.n]
                if axes:
                    for i in axes:
                        if i < 0:
                            i = i + len(output_shape)
                        output_shape.pop(i)
                else:
                    output_shape = [1]

            out = runtime.cinn_buffer_t(
                np.zeros(output_shape).astype("float32"),
                runtime.cinn_x86_device,
            )
            self.reduction_data = [
                x_data,
                x,
                out,
                runtime.cinn_pod_value_t(x),
                runtime.cinn_pod_value_t(out),
            ]

        return self.reduction_data


if __name__ == "__main__":
    unittest.main()
