#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from paddle import _C_ops

from .. import core
from ..framework import convert_np_dtype_to_dtype_

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DTypeLike

_supported_int_dtype_ = [
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
    core.VarDesc.VarType.BOOL,
]

# NOTE(chenweihang): We currently do not fully support the type promotion
# between tensors. Parting support here is because the interoperation of
# real and complex numbers in paddle quantum is very frequent, such as the
# binary operation between `float` and `complex64`, so we must support the
# correct type promotion on the APIs paddle quantum used.
# Now only check in dygraph (paddle quantum based dygraph)
# Full type promotion support will need to be fully verified later.
_supported_promote_complex_types_ = [
    '__add__',
    '__radd__',
    '__sub__',
    '__rsub__',
    '__mul__',
    '__rmul__',
    '__div__',
    '__truediv__',
    '__rdiv__',
    '__rtruediv__',
    '__matmul__',
]

_complex_dtypes = [
    core.VarDesc.VarType.COMPLEX64,
    core.VarDesc.VarType.COMPLEX128,
]

_already_patch_eager_tensor = False


def monkey_patch_math_tensor():
    """
    Similar to monkey_patch_variable.
    The difference is, in dygraph mode, use auto-generated op functions for better performance.
    """

    def astype(self: Tensor, dtype: DTypeLike) -> Tensor:
        """

        Cast a Tensor to a specified data type if it differs from the current dtype;
        otherwise, return the original Tensor.

        Args:
            dtype: The target data type.

        Returns:
            Tensor: a new Tensor with target dtype

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import numpy as np

                >>> original_tensor = paddle.ones([2, 2])
                >>> print("original tensor's dtype is: {}".format(original_tensor.dtype))
                original tensor's dtype is: paddle.float32
                >>> new_tensor = original_tensor.astype('float32')
                >>> print("new tensor's dtype is: {}".format(new_tensor.dtype))
                new tensor's dtype is: paddle.float32
        """
        if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
            dtype = convert_np_dtype_to_dtype_(dtype)

        if self.dtype == dtype:
            return self

        return _C_ops.cast(self, dtype)

    def _scalar_elementwise_op_(
        var: Tensor, scale: float, bias: float
    ) -> Tensor:
        return _C_ops.scale(var, float(scale), bias, True)

    def _neg_(var: Tensor) -> Tensor:
        return _scalar_elementwise_op_(var, -1.0, 0.0)

    def _abs_(var: Tensor) -> Tensor:
        return var.abs()

    def _complex_(var: Tensor) -> complex:
        numel = np.prod(var.shape)
        assert (
            numel == 1
        ), "only one element variable can be converted to complex."
        assert var._is_initialized(), "variable's tensor is not initialized"
        if not var.is_complex():
            var = var.astype('complex64')
        return complex(np.array(var))

    def _float_(var: Tensor) -> float:
        numel = np.prod(var.shape)
        assert (
            numel == 1
        ), "only one element variable can be converted to float."
        assert var._is_initialized(), "variable's tensor is not initialized"
        if (
            var.dtype == core.VarDesc.VarType.BF16
            or var.dtype == core.DataType.BFLOAT16
        ):
            var = var.astype('float32')
        return float(np.array(var))

    def _long_(var: Tensor) -> int:
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to long."
        assert var._is_initialized(), "variable's tensor is not initialized"
        if (
            var.dtype == core.VarDesc.VarType.BF16
            or var.dtype == core.DataType.BFLOAT16
        ):
            var = var.astype('float32')
        return int(np.array(var))

    def _int_(var: Tensor) -> int:
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to int."
        assert var._is_initialized(), "variable's tensor is not initialized"
        if (
            var.dtype == core.VarDesc.VarType.BF16
            or var.dtype == core.DataType.BFLOAT16
        ):
            var = var.astype('float32')
        return int(np.array(var))

    def _len_(var: Tensor) -> int:
        assert var.ndim > 0, "len() of a 0-D tensor is wrong"
        if var.type == core.VarDesc.VarType.VOCAB:
            return len(var.value().get_map_tensor())
        elif var.type == core.VarDesc.VarType.STRINGS:
            return len(var.value().get_string_tensor())
        else:
            return var.shape[0]

    def _index_(var: Tensor) -> int:
        numel = np.prod(var.shape)
        assert (
            numel == 1
        ), "only one element variable can be converted to python index."
        assert var._is_initialized(), "variable's tensor is not initialized"
        if (
            var.dtype == core.VarDesc.VarType.BF16
            or var.dtype == core.DataType.BFLOAT16
        ):
            var = var.astype('float32')
        return int(np.array(var))

    @property
    def _ndim(var: Tensor) -> int:
        return len(var.shape)

    def ndimension(var: Tensor) -> int:
        return len(var.shape)

    def dim(var: Tensor) -> int:
        return len(var.shape)

    @property
    def _size_(var: Tensor) -> int:
        return int(np.prod(var.shape))

    @property
    def _T_(var: Tensor) -> Tensor:
        if len(var.shape) == 1:
            return var
        perm = list(reversed(range(len(var.shape))))
        out = _C_ops.transpose(var, perm)
        return out

    @property
    def _mT_(var: Tensor) -> Tensor:
        if len(var.shape) < 2:
            raise ValueError(
                f"Tensor.ndim({var.ndim}) is required to be greater than or equal to 2."
            )
        perm = list(range(len(var.shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        out = _C_ops.transpose(var, perm)
        return out

    eager_methods = [
        ('__neg__', _neg_),
        ('__abs__', _abs_),
        ('__complex__', _complex_),
        ('__float__', _float_),
        ('__long__', _long_),
        ('__int__', _int_),
        ('__len__', _len_),
        ('__index__', _index_),
        ('astype', astype),
        ('dim', dim),
        ('ndimension', ndimension),
        ('ndim', _ndim),
        ('size', _size_),
        ('T', _T_),
        ('mT', _mT_),
        # for logical compare
        ('__array_ufunc__', None),
    ]

    eager_cpp_level_patch = [
        "__add__",
        "__radd__",
        '__sub__',
        '__rsub__',
        '__mul__',
        '__rmul__',
        '__div__',
        '__truediv__',
        '__rdiv__',
        '__rtruediv__',
        '__mod__',
        '__rmod__',
        '__matmul__',
        '__rmatmul__',
        '__gt__',
        '__ge__',
        '__lt__',
        '__le__',
        '__floordiv__',
        '__rfloordiv__',
        '__pow__',
        '__rpow__',
        '__eq__',
        '__ne__',
    ]

    global _already_patch_eager_tensor

    local_already_patch = _already_patch_eager_tensor
    _already_patch_eager_tensor = True
    local_tensor = core.eager.Tensor

    if not local_already_patch:
        for method_name in eager_cpp_level_patch:
            method_impl = getattr(local_tensor, method_name, None)
            if method_impl:
                setattr(local_tensor, method_name, method_impl)

        for method in eager_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(local_tensor, method_name, method_impl)
    else:
        import paddle.tensor

        # Tensor method from module paddle.tensor
        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(local_tensor, method_name):
                continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl:
                setattr(local_tensor, method_name, method_impl)

        for magic_method, origin_method in paddle.tensor.magic_method_func:
            impl = getattr(paddle.tensor, origin_method, None)
            if impl:
                setattr(local_tensor, magic_method, impl)
