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

from .. import core
from ..framework import (
    Variable,
    convert_np_dtype_to_dtype_,
    in_dygraph_mode,
)
from ..framework import _create_tensor as framework_create_tensor
from ..layers.layer_function_generator import OpProtoHolder
from . import no_grad
from .. import framework

import numpy as np
import warnings
from paddle import _C_ops, _legacy_C_ops

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

    def astype(self, dtype):
        """

        Cast a Tensor to a specified data type.

        Args:
            dtype: The target data type.

        Returns:
            Tensor: a new Tensor with target dtype

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                original_tensor = paddle.ones([2, 2])
                print("original tensor's dtype is: {}".format(original_tensor.dtype))
                new_tensor = original_tensor.astype('float32')
                print("new tensor's dtype is: {}".format(new_tensor.dtype))

        """
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        return _C_ops.cast(self, dtype)

    def _scalar_elementwise_op_(var, scale, bias):
        if framework.in_dygraph_mode():
            return _C_ops.scale(var, float(scale), bias, True)
        else:
            return _legacy_C_ops.scale(var, 'scale', scale, 'bias', bias)

    def _neg_(var):
        return _scalar_elementwise_op_(var, -1.0, 0.0)

    def _float_(var):
        numel = np.prod(var.shape)
        assert (
            numel == 1
        ), "only one element variable can be converted to float."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return float(np.array(var).flatten()[0])

    def _long_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to long."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return int(np.array(var).flatten()[0])

    def _int_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to int."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return int(np.array(var).flatten()[0])

    def _len_(var):
        assert var.ndim > 0, "len() of a 0-D tensor is wrong"
        if var.type == core.VarDesc.VarType.VOCAB:
            return len(var.value().get_map_tensor())
        elif var.type == core.VarDesc.VarType.STRINGS:
            return len(var.value().get_string_tensor())
        else:
            return var.shape[0]

    def _index_(var):
        numel = np.prod(var.shape)
        assert (
            numel == 1
        ), "only one element variable can be converted to python index."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return int(np.array(var).flatten()[0])

    @property
    def _ndim_(var):
        return len(var.shape)

    @property
    def _size_(var):
        return int(np.prod(var.shape))

    @property
    def _T_(var):
        if len(var.shape) == 1:
            return var
        perm = []
        for i in range(len(var.shape)):
            perm.insert(0, i)
        out = _C_ops.transpose(var, perm)
        return out

    eager_methods = [
        ('__neg__', _neg_),
        ('__float__', _float_),
        ('__long__', _long_),
        ('__int__', _int_),
        ('__len__', _len_),
        ('__index__', _index_),
        ('astype', astype),
        ('dim', lambda x: len(x.shape)),
        ('ndimension', lambda x: len(x.shape)),
        ('ndim', _ndim_),
        ('size', _size_),
        ('T', _T_),
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
        '__matmul__',
        '__gt__',
        '__ge__',
        '__lt__',
        '__le__',
        '__floordiv__',
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
