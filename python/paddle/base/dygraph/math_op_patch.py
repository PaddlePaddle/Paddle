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

import paddle
from paddle import _C_ops
from paddle.utils.decorator_utils import (
    size_args_decorator_patch,
)

from .. import core
from ..framework import convert_np_dtype_to_dtype_

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DTypeLike, PlaceLike, ShapeLike

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


_supported_dtype_conversions = {
    # float
    'float16': 'float16',
    'half': 'float16',
    'bfloat16': 'bfloat16',
    'float32': 'float32',
    'float': 'float32',
    'float64': 'float64',
    'double': 'float64',
    # int
    'int8': 'int8',
    'char': 'int8',
    # We handle uint8 conversion separately
    # 'uint8': 'uint8',
    # 'byte': 'uint8',
    'int16': 'int16',
    'short': 'int16',
    'int32': 'int32',
    'int': 'int32',
    'int64': 'int64',
    'long': 'int64',
    # other
    'bool': 'bool',
    'complex64': 'complex64',
    'complex128': 'complex128',
    'cfloat': 'complex64',
    'cdouble': 'complex128',
}


def monkey_patch_math_tensor():
    """
    Similar to monkey_patch_variable.
    The difference is, in dygraph mode, use auto-generated op functions for better performance.
    """
    global paddle

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

    def byte(self: Tensor) -> Tensor:
        # since paddle don't support float to uint8, so we need to convert it to int8 first
        if self.is_floating_point():
            tensor = astype(self, 'int8')
            return astype(tensor, 'uint8')
        elif self.is_complex():
            real = astype(self.real(), 'int8')
            return astype(real, 'uint8')
        else:
            return astype(self, 'uint8')

    def _create_dtype_conversion_methods():
        """
        Batch create all data type conversion methods
        """
        methods = []

        for method_name, target_dtype in _supported_dtype_conversions.items():

            def make_conversion_method(dtype):
                def conversion_method(self: Tensor) -> Tensor:
                    return astype(self, dtype)

                return conversion_method

            method_impl = make_conversion_method(target_dtype)
            method_impl.__name__ = method_name
            method_impl.__doc__ = f"""
            Cast a Tensor to {target_dtype} data type if it differs from the current dtype;
            otherwise, return the original Tensor.
            Returns:
                Tensor: a new Tensor with {target_dtype} dtype
            """

            methods.append((method_name, method_impl))

        return methods

    def type_as(self: Tensor, other: Tensor) -> Tensor:
        return self.astype(other.dtype)

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
        assert numel == 1, (
            "only one element variable can be converted to complex."
        )
        assert var._is_initialized(), "variable's tensor is not initialized"
        if not var.is_complex():
            var = var.astype('complex64')
        return complex(np.array(var))

    def _float_(var: Tensor) -> float:
        numel = np.prod(var.shape)
        assert numel == 1, (
            "only one element variable can be converted to float."
        )
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
        assert numel == 1, (
            "only one element variable can be converted to python index."
        )
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

    def _new_full_(
        var: Tensor,
        size: ShapeLike,
        fill_value: bool | float | paddle.Tensor,
        *,
        dtype: DTypeLike | None = None,
        device: PlaceLike | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> Tensor:
        if dtype is None:
            dtype = var.dtype
        if device is None:
            device = var.place

        return paddle.full(
            size,
            fill_value,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )

    @size_args_decorator_patch
    def _new_empty_(
        var: Tensor,
        size: ShapeLike,
        *,
        dtype: DTypeLike | None = None,
        device: PlaceLike | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> Tensor:
        if dtype is None:
            dtype = var.dtype
        if device is None:
            device = var.place

        return paddle.empty(
            size,
            dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )

    @size_args_decorator_patch
    def _new_ones_(
        var: Tensor,
        size: ShapeLike,
        *,
        dtype: DTypeLike | None = None,
        device: PlaceLike | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> Tensor:
        if dtype is None:
            dtype = var.dtype
        if device is None:
            device = var.place

        return paddle.full(
            size,
            1,
            dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )

    @size_args_decorator_patch
    def _new_zeros_(
        var: Tensor,
        size: ShapeLike,
        *,
        dtype: DTypeLike | None = None,
        device: PlaceLike | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> Tensor:
        if dtype is None:
            dtype = var.dtype
        if device is None:
            device = var.place

        return paddle.full(
            size,
            0,
            dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )

    @property
    def requires_grad(self: Tensor) -> bool:
        """
        Whether this Tensor requires gradient computation.

        This is a convenience property that returns the opposite of stop_gradient.
        Setting requires_grad=True is equivalent to setting stop_gradient=False.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> x = paddle.randn([2, 3])
                >>> print(x.requires_grad)  # False by default
                >>>
                >>> x.requires_grad = False
                >>> print(x.stop_gradient)  # True
        """
        return not self.stop_gradient

    @requires_grad.setter
    def requires_grad(self: Tensor, value: bool) -> None:
        """
        Set whether this Tensor requires gradient computation.

        Args:
            value (bool): True to enable gradient computation, False to disable.
        """
        if not isinstance(value, bool):
            raise TypeError(
                f"requires_grad must be bool, but got {type(value)}"
            )
        self.stop_gradient = not value

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
        ('byte', byte),
        ('uint8', byte),
        ('type_as', type_as),
        ('dim', dim),
        ('ndimension', ndimension),
        ('ndim', _ndim),
        ('size', _size_),
        ('T', _T_),
        ('mT', _mT_),
        ('new_full', _new_full_),
        ('new_empty', _new_empty_),
        ('new_ones', _new_ones_),
        ('new_zeros', _new_zeros_),
        ("requires_grad", requires_grad),
        # for logical compare
        ('__array_ufunc__', None),
    ]

    dtype_conversion_methods = _create_dtype_conversion_methods()
    eager_methods.extend(dtype_conversion_methods)

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
