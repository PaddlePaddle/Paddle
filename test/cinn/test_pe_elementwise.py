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
import scipy
from cinn import Target, ir, lang, pe, runtime
from cinn.poly import create_stages


class TestPEElementwise(unittest.TestCase):
    def setUp(self):
        self.m = 32
        self.n = 32

        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k32
        self.target.os = Target.OS.Linux
        cinn.set_target(self.target)
        self.unary_data = []

    def test_unary(self):
        for fn_name, pe_fn, np_fn, dtype, low, high in [
            ("exp", pe.exp, np.exp, "float32", -10, 10),
            ("erf", pe.erf, scipy.special.erf, "float32", -99, 99),
            ("sqrt", pe.sqrt, np.sqrt, "float32", 0.1, 10),
            ("log", pe.log, np.log, "float32", 0.1, 99),
            ("log2", pe.log2, np.log2, "float32", 0.1, 99),
            ("log10", pe.log10, np.log10, "float32", 0.1, 99),
            ("floor", pe.floor, np.floor, "float32", -99, 99),
            ("ceil", pe.ceil, np.ceil, "float32", -99, 99),
            ("round", pe.round, np.round, "float32", -99, 99),
            ("trunc", pe.trunc, np.trunc, "float32", -99, 99),
            ("cos", pe.cos, np.cos, "float32", -2.0 * np.pi, 2.0 * np.pi),
            ("cosh", pe.cosh, np.cosh, "float32", -2.0 * np.pi, 2.0 * np.pi),
            ("tan", pe.tan, np.tan, "float32", -2.0 * np.pi, 2.0 * np.pi),
            ("tanh", pe.tanh, np.tanh, "float32", -2.0 * np.pi, 2.0 * np.pi),
            ("tanh", pe.tanh, np.tanh, "float32", -2.0 * np.pi, 2.0 * np.pi),
            ("sin", pe.sin, np.sin, "float32", -2.0 * np.pi, 2.0 * np.pi),
            ("sinh", pe.sinh, np.sinh, "float32", -2.0 * np.pi, 2.0 * np.pi),
            # TODO(wenming2014) not numpy
            # ("acos", pe.acos, np.acos, "float32", -99, 99),
            # ("acosh", pe.acosh, np.acosh, "float32"),
            # ("asin", pe.asin, np.asin, "float32"),
            # ("asinh", pe.asinh, np.asinh, "float32"),
            # ("atan", pe.atan, np.atan, "float32"),
            # ("atanh", pe.atanh, np.atanh, "float32"),
            ("isnan", pe.isnan, np.isnan, "float32", -99, 99),
            ("isfinite", pe.isfinite, np.isfinite, "float32", -99, 99),
            ("isinf", pe.isinf, np.isinf, "float32", -99, 99),
            ("negative", pe.negative, np.negative, "float32", -99, 99),
            # TODO(wenming2014) further support
            # ("identity", pe.identity, np.identity, "float32",-99,99),
            # ("logical_not", pe.logical_not, np.logical_not, "bool",0,1),
            ("bitwise_not", pe.bitwise_not, np.bitwise_not, "int32", -99, 99),
            (
                "sigmoid",
                pe.sigmoid,
                lambda x: 1 / (1 + np.exp(-x)),
                "float32",
                -99,
                99,
            ),
            ("sign", pe.sign, np.sign, "float32", -99, 99),
            ("abs", pe.abs, np.abs, "float32", -99, 99),
            (
                "rsqrt",
                pe.rsqrt,
                lambda x: np.ones_like(x) / np.sqrt(x),
                "float32",
                0.1,
                99,
            ),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            is_round = fn_name == "round"
            is_bool = (
                (fn_name == "isnan")
                | (fn_name == "isfinite")
                | (fn_name == "isinf")
                | (fn_name == "logical_not")
            )
            self.union_tester(
                fn_name, pe_fn, np_fn, dtype, low, high, is_round, is_bool
            )

    def union_tester(
        self,
        fn_name,
        cinn_fn,
        np_fn,
        dtype="float32",
        low=0,
        high=1,
        is_round=False,
        is_bool=False,
    ):
        m, n = (
            ir.Expr(_)
            for _ in (
                self.m,
                self.n,
            )
        )

        x = lang.Placeholder(dtype, "x", [m, n])
        y = cinn_fn(x.to_tensor())

        func_name = "test_" + fn_name

        args = [x.to_tensor()]
        for out in y:
            args.append(out)
        stages = create_stages(args)
        func = lang.lower(func_name, stages, args)

        builder = lang.Module.Builder("elementwise_module", self.target)
        builder.add_function(func)

        module = builder.build()
        self.compiler.build(module)

        fn = self.compiler.lookup(func_name)

        x_data, x_buf, out_buf, *args = self.create_data(
            dtype, low, high, is_round, is_bool
        )
        fn(args)

        self.assertTrue(
            np.allclose(
                out_buf.numpy(),
                self.create_target_data(x_data, np_fn),
                atol=1e-4,
            ),
            func_name,
        )

    def create_target_data(self, x_data, np_target_fn):
        return np_target_fn(x_data)

    def create_data(self, dtype, low, high, is_round, is_bool):
        self.unary_data.clear()
        if not self.unary_data:
            x_data = np.around(
                np.random.uniform(low, high, (self.m, self.n)).astype(dtype), 2
            )
            if is_round:
                x_data += ((np.abs(np.fmod(x_data, 1)) - 0.5) < 1e-6) * 1e-4
            x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
            if is_bool:
                out = runtime.cinn_buffer_t(
                    np.zeros([self.m, self.n]).astype(np.bool_),
                    runtime.cinn_x86_device,
                )
            else:
                out = runtime.cinn_buffer_t(
                    np.zeros([self.m, self.n]).astype(dtype),
                    runtime.cinn_x86_device,
                )
            self.unary_data = [
                x_data,
                x,
                out,
                runtime.cinn_pod_value_t(x),
                runtime.cinn_pod_value_t(out),
            ]

        return self.unary_data


if __name__ == "__main__":
    unittest.main()
