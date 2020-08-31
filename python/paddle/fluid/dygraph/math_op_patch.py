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
from ..framework import Variable, convert_np_dtype_to_dtype_, _varbase_creator
from ..layers.layer_function_generator import OpProtoHolder
from ..layers import common_methods
from . import to_variable, no_grad
import paddle

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

    @no_grad()
    def create_tensor(value, dtype, shape):
        out = _varbase_creator(dtype=dtype)
        out = core.ops.fill_constant(out, 'dtype', dtype, 'shape', shape,
                                     'value', value, 'force_cpu', False)
        out.stop_gradient = True
        return out

    def create_scalar(value, dtype):
        return create_tensor(value, dtype, shape=[1])

    def astype(self, dtype):
        """
        **Notes**:
            **The variable must be a** :ref:`api_fluid_Tensor`

        Cast a variable to a specified data type.

        Args:

            self(Variable): The source variable

            dtype: The target data type

        Returns:
            Variable: Variable with new dtype

        Examples:
            In Static Graph Mode:

            .. code-block:: python

                import paddle.fluid as fluid

                startup_prog = fluid.Program()
                main_prog = fluid.Program()
                with fluid.program_guard(startup_prog, main_prog):
                    original_variable = fluid.data(name = "new_variable", shape=[2,2], dtype='float32')
                    new_variable = original_variable.astype('int64')
                    print("new var's dtype is: {}".format(new_variable.dtype))

            In Dygraph Mode:

            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    original_variable = fluid.dygraph.to_variable(x)
                    print("original var's dtype is: {}, numpy dtype is {}".format(original_variable.dtype, original_variable.numpy().dtype))
                    new_variable = original_variable.astype('int64')
                    print("new var's dtype is: {}, numpy dtype is {}".format(new_variable.dtype, new_variable.numpy().dtype))

        """
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        return core.ops.cast(self, 'in_dtype', self.dtype, 'out_dtype', dtype)

    def _scalar_elementwise_op_(var, scale, bias):
        return core.ops.scale(var, 'scale', scale, 'bias', bias)

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

    def _scalar_add_(var, value):
        return _scalar_elementwise_op_(var, 1.0, value)

    def _scalar_sub_(var, value):
        return _scalar_elementwise_op_(var, 1.0, -value)

    def _scalar_rsub_(var, value):
        return _scalar_elementwise_op_(var, -1.0, value)

    def _scalar_mul_(var, value):
        return _scalar_elementwise_op_(var, value, 0.0)

    def _scalar_div_(var, value):
        return _scalar_elementwise_op_(var, 1.0 / value, 0.0)

    # TODO(shenliang03):  currently, it supports divide, floor_divide, remainder
    # for binary operator by using the api to achieve the type promotion
    def _binary_method_creator_(op_type, reverse=False):
        import paddle

        def __impl__(self, other_var):
            import paddle
            op = getattr(paddle, op_type)
            if reverse:
                return op(other_var, self)
            else:
                return op(self, other_var)

        __impl__.__doc__ = """

        See paddle.{}""".format(op_type)
        __impl__.__name__ = op_type

        return __impl__

    # for binary operator such as elementwise, compare
    def _binary_creator_(method_name,
                         op_type,
                         reverse=False,
                         scalar_method=None):
        def __impl__(self, other_var):
            # FIXME(zjl): elementwise_div between integers cannot be converted to scale,
            # which may lose accuracy. This is a hot fix for release 1.6.
            if scalar_method is not None and not (
                    op_type == 'elementwise_div' and
                    self.dtype in _supported_int_dtype_):
                if isinstance(other_var, float):
                    if self.dtype in _supported_int_dtype_:
                        assert other_var == int(other_var), \
                            "float value {} cannot convert to integer".format(other_var)
                    return scalar_method(self, other_var)
                elif isinstance(other_var, int):
                    return scalar_method(self, float(other_var))

            lhs_dtype = self.dtype

            if not isinstance(other_var, core.VarBase):
                if reverse:
                    other_var = create_tensor(
                        other_var, dtype=lhs_dtype, shape=self.shape)
                else:
                    # add fill_op 
                    other_var = create_scalar(value=other_var, dtype=lhs_dtype)

            rhs_dtype = other_var.dtype
            if lhs_dtype != rhs_dtype:
                other_var = astype(other_var, lhs_dtype)
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
            self(Tensor): left hand Tensor
            other_var(Tensor|float|int): right hand Tensor

        Returns:
            Tensor
        """.format(comment)
        __impl__.__name__ = method_name
        return __impl__

    # Todo(zhouwei): implement dygraph template to adapt to any function, receive('op_type', 'arg_template')
    #  Such as _method_creator_('addmm', 'x, y, alpha=1.0, beta=1.0, name=None'). It can reduce call time.
    def _method_creator_(op_type, arg_template=None):
        def __impl__(self):
            op = getattr(core.ops, op_type)
            return op(self)

        __impl__.__doc__ = """

        See paddle.{}""".format(op_type)
        __impl__.__name__ = op_type

        return __impl__

    varbase_methods = [
        # Type1: From custom fun or lambda
        ##   b=-a
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
        ('size', lambda x: x.shape),
        # Type2: From Template that create core.ops automatically. It's recommended.
        ('__add__',
         _binary_creator_('__add__', 'elementwise_add', False, _scalar_add_)),
        ##  a+b == b+a. Do not need to reverse explicitly
        ('__radd__',
         _binary_creator_('__radd__', 'elementwise_add', False, _scalar_add_)),
        ('__sub__', _binary_creator_('__sub__', 'elementwise_sub', False,
                                     _scalar_sub_)),
        ('__rsub__', _binary_creator_('__rsub__', 'elementwise_sub', True,
                                      _scalar_rsub_)),
        ('__mul__', _binary_creator_('__mul__', 'elementwise_mul', False,
                                     _scalar_mul_)),
        ## a*b == b*a. Do not need to reverse explicitly
        ('__rmul__',
         _binary_creator_('__rmul__', 'elementwise_mul', False, _scalar_mul_)),
        ('__rtruediv__', _binary_creator_('rtruediv__', 'elementwise_div', True,
                                          None)),
        ('__pow__', _binary_creator_('__pow__', 'elementwise_pow', False,
                                     None)),
        ('__rpow__', _binary_creator_('__rpow__', 'elementwise_pow', True,
                                      None)),
        # These binary use paddle.optype
        ('__div__', _binary_method_creator_('divide', False)),
        ('__truediv__', _binary_method_creator_('divide', False)),
        ('__rtruediv__', _binary_method_creator_('divide', True)),
        ('__rdiv__', _binary_method_creator_('divide', True)),
        ('__floordiv__', _binary_method_creator_('floor_divide', False)),
        ('__rfloordiv__', _binary_method_creator_('floor_divide', True)),
        ('__mod__', _binary_method_creator_('remainder', False)),
        ## for logical compare
        ('__eq__', _binary_creator_('__eq__', 'equal', False, None)),
        ('__ne__', _binary_creator_('__ne__', 'not_equal', False, None)),
        ('__lt__', _binary_creator_('__lt__', 'less_than', False, None)),
        ('__le__', _binary_creator_('__le__', 'less_equal', False, None)),
        ('__gt__', _binary_creator_('__gt__', 'greater_than', False, None)),
        ('__ge__', _binary_creator_('__ge__', 'greater_equal', False, None)),
        ('__array_ufunc__', None),
        ('sigmoid', _method_creator_('sigmoid', 'name=None')),
        ('logsigmoid', _method_creator_('logsigmoid', 'name=None')),
        ('exp', _method_creator_('exp', 'name=None')),
        ('tanh', _method_creator_('tanh', 'name=None')),
        ('atan', _method_creator_('atan', 'name=None')),
        ('tanh_shrink', _method_creator_('tanh_shrink', 'name=None')),
        ('sqrt', _method_creator_('sqrt', 'name=None')),
        ('rsqrt', _method_creator_('rsqrt', 'name=None')),
        ('abs', _method_creator_('abs', 'name=None')),
        ('ceil', _method_creator_('ceil', 'name=None')),
        ('floor', _method_creator_('floor', 'name=None')),
        ('cos', _method_creator_('cos', 'name=None')),
        ('acos', _method_creator_('acos', 'name=None')),
        ('asin', _method_creator_('asin', 'name=None')),
        ('sin', _method_creator_('sin', 'name=None')),
        ('sinh', _method_creator_('sinh', 'name=None')),
        ('cosh', _method_creator_('cosh', 'name=None')),
        ('round', _method_creator_('round', 'name=None')),
        ('reciprocal', _method_creator_('reciprocal', 'name=None')),
        ('square', _method_creator_('square', 'name=None')),
        ('softplus', _method_creator_('softplus', 'name=None')),
        ('softsign', _method_creator_('softsign', 'name=None')),
        # Type3: Form module 'paddle.tensor' defaultly.
        #   It's not a goodway, because it will increase call time.
    ]

    global _already_patch_varbase
    if not _already_patch_varbase:
        for method in varbase_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(core.VarBase, method_name, method_impl)
    else:
        import paddle.tensor
        for method_name in common_methods:
            if hasattr(core.VarBase, method_name): continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl: setattr(core.VarBase, method_name, method_impl)

    _already_patch_varbase = True
