# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
Tests that static_api_run_custom_op correctly handles pir::Value attributes
(produced by pd_op.full) in PIR/dy2st mode.

The fix (commit aa837b6) adds pir::Value handling for five scalar attr types
in manual_static_op_function.h: bool, int, float, double, int64_t.

Note on double in PIR mode:
  CppTypeToAttrTypeMap() in op_dialect.cc does not include "double", so
  custom ops with double attrs cannot be registered in PIR mode at all.
  The double branch in manual_static_op_function.h is therefore tested in
  eager mode only via a dedicated op (custom_double_attr).
  The four PIR-compatible types (bool/int/float/int64_t) are tested both in
  eager mode and in PIR/dy2st mode.
"""

import os
import textwrap
import unittest

import numpy as np
from utils import extra_cc_args, extra_nvcc_args, paddle_includes

import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

_build_dir = os.path.join(get_build_directory(), 'custom_scalar_attr_pir')
_pyd = os.path.join(_build_dir, 'custom_scalar_attr_pir.pyd')
if os.name == 'nt' and os.path.isfile(_pyd):
    run_cmd(f'del {_pyd}', True)

# ---------------------------------------------------------------------------
# Op 1: four PIR-compatible scalar attr types (bool, int, float, int64_t)
# ---------------------------------------------------------------------------
_pir_op_source = textwrap.dedent("""\
    #include "paddle/extension.h"

    // Kernel: scale x.
    // use_int64  - if true, use long_scale cast to float instead of scale_f32
    // max_len    - int guard (must be > 0)
    // scale_f32  - float scale used when use_int64 == false
    // long_len   - int64_t guard (must be > 0); also serves as scale source
    std::vector<paddle::Tensor> PirAttrsForward(
        const paddle::Tensor& x,
        bool    use_int64,
        int     max_len,
        float   scale_f32,
        int64_t long_len) {
      PADDLE_ENFORCE_GT(max_len,  0, phi::errors::InvalidArgument(
          "max_len must be > 0, got %d", max_len));
      PADDLE_ENFORCE_GT(long_len, 0, phi::errors::InvalidArgument(
          "long_len must be > 0, got %lld", (long long)long_len));
      float effective_scale = use_int64
          ? static_cast<float>(long_len)
          : scale_f32;
      auto out = paddle::experimental::scale(x, effective_scale, 0.0f, true);
      return {out};
    }

    std::vector<std::vector<int64_t>> PirAttrsInferShape(
        std::vector<int64_t> x_shape,
        bool use_int64, int max_len, float scale_f32, int64_t long_len) {
      return {x_shape};
    }

    std::vector<paddle::DataType> PirAttrsInferDtype(
        paddle::DataType x_dtype,
        bool use_int64, int max_len, float scale_f32, int64_t long_len) {
      return {x_dtype};
    }

    PD_BUILD_OP(custom_pir_scalar_attrs)
        .Inputs({"X"})
        .Outputs({"Out"})
        .Attrs({"use_int64: bool",
                "max_len: int",
                "scale_f32: float",
                "long_len: int64_t"})
        .SetKernelFn(PD_KERNEL(PirAttrsForward))
        .SetInferShapeFn(PD_INFER_SHAPE(PirAttrsInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(PirAttrsInferDtype));
""")

# ---------------------------------------------------------------------------
# Op 2: double attr - eager mode only (double is unsupported in PIR custom op
# registration: not present in CppTypeToAttrTypeMap in op_dialect.cc).
# The double branch in manual_static_op_function.h is still tested via eager.
# ---------------------------------------------------------------------------
_double_op_source = textwrap.dedent("""\
    #include "paddle/extension.h"

    std::vector<paddle::Tensor> DoubleAttrForward(
        const paddle::Tensor& x,
        double scale_f64) {
      auto out = paddle::experimental::scale(
          x, static_cast<float>(scale_f64), 0.0f, true);
      return {out};
    }

    std::vector<std::vector<int64_t>> DoubleAttrInferShape(
        std::vector<int64_t> x_shape, double scale_f64) {
      return {x_shape};
    }

    // NOTE: InferDtypeCallHelper has no specialization for double
    // (op_meta_info.h), so SetInferDtypeFn is intentionally omitted.
    PD_BUILD_OP(custom_double_attr)
        .Inputs({"X"})
        .Outputs({"Out"})
        .Attrs({"scale_f64: double"})
        .SetKernelFn(PD_KERNEL(DoubleAttrForward))
        .SetInferShapeFn(PD_INFER_SHAPE(DoubleAttrInferShape));
""")

os.makedirs(_build_dir, exist_ok=True)

_pir_src = os.path.join(_build_dir, 'custom_pir_scalar_attrs.cc')
_double_src = os.path.join(_build_dir, 'custom_double_attr.cc')
for path, src in [(_pir_src, _pir_op_source), (_double_src, _double_op_source)]:
    with open(path, 'w') as f:
        f.write(src)

custom_module = load(
    name='custom_scalar_attr_pir',
    sources=[_pir_src, _double_src],
    extra_include_paths=paddle_includes,
    extra_cxx_cflags=extra_cc_args,
    extra_cuda_cflags=extra_nvcc_args,
    build_directory=_build_dir,
    verbose=True,
)

_pir_op = custom_module.custom_pir_scalar_attrs
_double_op = custom_module.custom_double_attr


# ---------------------------------------------------------------------------
# Eager baseline tests: all five scalar attr types with Python literals
# ---------------------------------------------------------------------------
class TestCustomScalarAttrEager(unittest.TestCase):
    """Baseline: scalar attrs passed as plain Python literals in eager mode."""

    def setUp(self):
        paddle.disable_static()
        self.x = paddle.randn([2, 4], dtype='float32')

    def test_bool_false(self):
        # use_int64=False -> scale_f32=2.0 is used
        out = _pir_op(self.x, False, 512, 2.0, 1024)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 2.0).numpy(), rtol=1e-5
        )

    def test_bool_true(self):
        # use_int64=True -> long_len=3 cast to float is used as scale
        out = _pir_op(self.x, True, 512, 99.0, 3)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 3.0).numpy(), rtol=1e-5
        )

    def test_int_attr(self):
        out = _pir_op(self.x, False, 1, 0.5, 1)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 0.5).numpy(), rtol=1e-5
        )

    def test_float_attr(self):
        out = _pir_op(self.x, False, 1, 1.5, 1)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 1.5).numpy(), rtol=1e-5
        )

    def test_int64_attr(self):
        out = _pir_op(self.x, True, 1, 0.0, 4)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 4.0).numpy(), rtol=1e-5
        )

    def test_double_attr(self):
        # double attr: only supported in eager mode (not PIR).
        out = _double_op(self.x, 2.5)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 2.5).numpy(), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# PIR/dy2st tests: four supported types as pir::Value (from paddle.full)
# ---------------------------------------------------------------------------
class TestCustomScalarAttrPIR(unittest.TestCase):
    """
    Key tests: scalar attrs arrive as pir::Value in PIR/dy2st mode.

    paddle.full() inside to_static produces pd_op.full whose output is a
    pir::Value.  static_api_run_custom_op must call GetScalarFromPirValue to
    extract the constant and store it as the correct pir Attribute, instead
    of crashing with:
      "argument (position N) must be <type>, but got pir::Value"

    double is excluded: CppTypeToAttrTypeMap() in op_dialect.cc does not
    contain "double", so custom ops with double attrs cannot be registered
    in PIR at all (fails at op registration, not at the fixed code path).
    """

    def setUp(self):
        paddle.disable_static()
        self.x = paddle.randn([2, 4], dtype='float32')

    # ------------------------------------------------------------------
    # bool attr via pir::Value
    # ------------------------------------------------------------------
    def test_bool_false_pir_value(self):
        @paddle.jit.to_static(full_graph=True)
        def model_fn(x):
            use_int64 = paddle.full([], 0, dtype='bool')  # False
            return _pir_op(x, use_int64, 512, 2.0, 1024)

        out = model_fn(self.x)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 2.0).numpy(), rtol=1e-5
        )

    def test_bool_true_pir_value(self):
        @paddle.jit.to_static(full_graph=True)
        def model_fn(x):
            use_int64 = paddle.full([], 1, dtype='bool')  # True
            return _pir_op(x, use_int64, 512, 99.0, 3)

        out = model_fn(self.x)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 3.0).numpy(), rtol=1e-5
        )

    # ------------------------------------------------------------------
    # int attr via pir::Value
    # ------------------------------------------------------------------
    def test_int_pir_value_int32(self):
        @paddle.jit.to_static(full_graph=True)
        def model_fn(x):
            max_len = paddle.full([], 512, dtype='int32')
            return _pir_op(x, False, max_len, 2.0, 1024)

        out = model_fn(self.x)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 2.0).numpy(), rtol=1e-5
        )

    def test_int_pir_value_int64_cast(self):
        """int attr from int64 full (value fits in int32)."""

        @paddle.jit.to_static(full_graph=True)
        def model_fn(x):
            max_len = paddle.full([], 256, dtype='int64')
            return _pir_op(x, False, max_len, 0.5, 1024)

        out = model_fn(self.x)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 0.5).numpy(), rtol=1e-5
        )

    # ------------------------------------------------------------------
    # float attr via pir::Value
    # ------------------------------------------------------------------
    def test_float_pir_value(self):
        @paddle.jit.to_static(full_graph=True)
        def model_fn(x):
            scale_f32 = paddle.full([], 1.5, dtype='float32')
            return _pir_op(x, False, 512, scale_f32, 1024)

        out = model_fn(self.x)
        np.testing.assert_allclose(
            out.numpy(), (self.x * 1.5).numpy(), rtol=1e-5
        )

    # ------------------------------------------------------------------
    # int64_t attr via pir::Value
    # ------------------------------------------------------------------
    def test_int64_pir_value(self):
        @paddle.jit.to_static(full_graph=True)
        def model_fn(x):
            long_len = paddle.full([], 5, dtype='int64')
            return _pir_op(x, True, 512, 0.0, long_len)

        out = model_fn(self.x)
        # use_int64=True -> long_len=5 used as scale
        np.testing.assert_allclose(
            out.numpy(), (self.x * 5.0).numpy(), rtol=1e-5
        )

    # ------------------------------------------------------------------
    # All four PIR-compatible attrs as pir::Value simultaneously
    # ------------------------------------------------------------------
    def test_all_pir_attrs_as_value(self):
        @paddle.jit.to_static(full_graph=True)
        def model_fn(x):
            use_int64 = paddle.full([], 1, dtype='bool')
            max_len = paddle.full([], 512, dtype='int32')
            scale_f32 = paddle.full([], 99.0, dtype='float32')
            long_len = paddle.full([], 7, dtype='int64')
            return _pir_op(x, use_int64, max_len, scale_f32, long_len)

        out = model_fn(self.x)
        # use_int64=True -> long_len=7 used as scale
        np.testing.assert_allclose(
            out.numpy(), (self.x * 7.0).numpy(), rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
