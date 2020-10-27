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

from __future__ import print_function

from .. import core
from ..framework import Variable, convert_np_dtype_to_dtype_, _varbase_creator, _current_expected_place
from ..layers.layer_function_generator import OpProtoHolder
from . import no_grad

import numpy as np
import six

_supported_int_dtype_ = [
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
]

_already_patch_varbase = False


def monkey_patch_math_varbase():
    """
    Similar to monkey_patch_variable.
    The difference is, in dygraph mode, use auto-generated op functions for better performance.
    """

    @no_grad
    def create_tensor(value, is_scalar=False):
        out = core.VarBase(
            value=value,
            place=_current_expected_place(),
            persistable=False,
            zero_copy=True,
            stop_gradient=True,
            is_scalar=is_scalar)
        return out

    def create_scalar(value):
        return create_tensor([value], True)

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
        return core.ops.cast(self, 'in_dtype', self.dtype, 'out_dtype', dtype)

    def _neg_(var):
        return _scalar_elementwise_op_(var, -1.0, 0.0)

    def _float_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to float."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return float(var.numpy().flatten()[0])

    def _long_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to long."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        if six.PY2:
            return long(var.numpy().flatten()[0])
        else:
            return int(var.numpy().flatten()[0])

    def _int_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to int."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return int(var.numpy().flatten()[0])

    def _len_(var):
        return var.shape[0]

    def _index_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to python index."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        if six.PY2:
            return long(var.numpy().flatten()[0])
        else:
            return int(var.numpy().flatten()[0])

    @property
    def _ndim_(var):
        return len(var.shape)

    @property
    def _size_(var):
        return np.prod(var.shape)

    # for binary operator such as elementwise, compare
    def _binary_creator_(method_name, op_type, reverse=False):
        def __impl__(self, other_var):
            lhs_dtype = self.dtype

            if not isinstance(other_var, core.VarBase):
                if reverse:
                    other_var = create_tensor(other_var)
                else:
                    other_var = create_scalar(other_var)

            print("lhs_dtype: ", lhs_dtype)
            print("rhs_dtype: ", other_var.dtype)

            if reverse:
                tmp = self
                self = other_var
                other_var = tmp

            axis = -1
            math_op = getattr(core.ops, op_type)
            return math_op(self, other_var, 'axis', axis)

        comment = OpProtoHolder.instance().get_op_proto(op_type).comment

        __impl__.__doc__ = """
        {0}
        Args:
            other_var(Tensor|float|int): right hand Tensor

        Returns:
            Tensor
        """.format(comment)
        __impl__.__name__ = method_name
        return __impl__

    varbase_methods = [
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
        ('__add__', _binary_creator_('__add__', 'elementwise_add', False)),
        ##  a+b == b+a. Do not need to reverse explicitly
        ('__radd__', _binary_creator_('__radd__', 'elementwise_add', False)),
        ('__sub__', _binary_creator_('__sub__', 'elementwise_sub', False)),
        ('__rsub__', _binary_creator_('__rsub__', 'elementwise_sub', True)),
        ('__mul__', _binary_creator_('__mul__', 'elementwise_mul', False)),
        ## a*b == b*a. Do not need to reverse explicitly
        ('__rmul__', _binary_creator_('__rmul__', 'elementwise_mul', False)),
        ('__div__', _binary_creator_('__div__', 'elementwise_div', False)),
        ('__truediv__', _binary_creator_('__truediv__', 'elementwise_div',
                                         False)),
        ('__rdiv__', _binary_creator_('__rdiv__', 'elementwise_div', True)),
        ('__rtruediv__', _binary_creator_('rtruediv__', 'elementwise_div',
                                          True)),
        ('__pow__', _binary_creator_('__pow__', 'elementwise_pow', False)),
        ('__rpow__', _binary_creator_('__rpow__', 'elementwise_pow', True)),
        ('__floordiv__', _binary_creator_('__floordiv__',
                                          'elementwise_floordiv', False)),
        ('__mod__', _binary_creator_('__mod__', 'elementwise_mod', False)),
        ## for logical compare
        ('__eq__', _binary_creator_('__eq__', 'equal', False)),
        ('__ne__', _binary_creator_('__ne__', 'not_equal', False)),
        ('__lt__', _binary_creator_('__lt__', 'less_than', False)),
        ('__le__', _binary_creator_('__le__', 'less_equal', False)),
        ('__gt__', _binary_creator_('__gt__', 'greater_than', False)),
        ('__ge__', _binary_creator_('__ge__', 'greater_equal', False)),
        ('__array_ufunc__', None)
    ]

    global _already_patch_varbase
    if not _already_patch_varbase:
        for method in varbase_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(core.VarBase, method_name, method_impl)
    else:
        import paddle.tensor
        # Tensor method from module paddle.tensor
        tensor_methods = paddle.tensor.linalg.__all__ + \
                         paddle.tensor.math.__all__ + \
                         paddle.tensor.logic.__all__ + \
                         paddle.tensor.manipulation.__all__ + \
                         paddle.tensor.search.__all__ + \
                         paddle.tensor.stat.__all__ + \
                         paddle.tensor.attribute.__all__
        for method_name in tensor_methods:
            if hasattr(core.VarBase, method_name): continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl: setattr(core.VarBase, method_name, method_impl)

    _already_patch_varbase = True
