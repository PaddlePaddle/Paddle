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

import inspect
import warnings

from .. import core
from ..dygraph.base import in_to_static_mode
from ..framework import (
    OpProtoHolder,
    Variable,
    default_main_program,
    static_only,
)

_supported_int_dtype_ = [
    core.VarDesc.VarType.BOOL,
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
]
_supported_complex_dtype_ = [
    core.VarDesc.VarType.COMPLEX64,
    core.VarDesc.VarType.COMPLEX128,
]


compare_ops = ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']

SUPPORT_PROMOTION_OPS = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__mod__",
    "__rmod__",
    "__div__",
    "__rdiv__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__pow__",
    "__rpow__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
]

EXPRESSION_MAP = {
    "__add__": "A + B",
    "__radd__": "A += B",
    "__sub__": "A - B",
    "__rsub__": "A -= B",
    "__mul__": "A * B",
    "__rmul__": "A *= B",
    "__div__": "A / B",
    "__truediv__": "A / B",
    "__rdiv__": "A /= B",
    "__rtruediv__": "A /= B",
    "__pow__": "A ** B",
    "__rpow__": "A **= B",
    "__floordiv__": "A //B",
    "__rfloordiv__": "A //=B",
    "__mod__": "A % B",
    "__rmod__": "A %= B",
    "__matmul__": "A @ B",
    "__rmatmul__": "A @= B",
    "__eq__": "A == B",
    "__ne__": "A != B",
    "__lt__": "A < B",
    "__le__": "A <= B",
    "__gt__": "A > B",
    "__ge__": "A >= B",
}

_already_patch_variable = False


# TODO(liym27): A better way to slice tensor array.
#  Maybe support start == end for slice op.
def _slice_tensor_array(array, start, end):
    from paddle.static.nn import cond
    from paddle.tensor import create_array

    def true_fn():
        null_array = create_array("float32")
        return null_array

    def false_fn(array, start, end):
        new_array = array[start:end]
        return new_array

    new_array = cond(start == end, true_fn, lambda: false_fn(array, start, end))
    return new_array


def monkey_patch_variable():
    def unique_tmp_name():
        return default_main_program()._name_generator.generate("tmp")

    def safe_get_dtype(var):
        try:
            dtype = var.dtype
        except:
            raise ValueError(f"Cannot get data type from {var.name}")
        return dtype

    def current_block(var):
        return var.block.program.current_block()

    def create_new_tmp_var(block, dtype):
        tmp_name = unique_tmp_name()
        return block.create_var(name=tmp_name, dtype=dtype)

    def create_new_tmp_sparse_var(block, dtype, type):
        tmp_name = unique_tmp_name()
        return block.create_var(name=tmp_name, dtype=dtype, type=type)

    def create_tensor(block, value, dtype, shape):
        value = float(value)
        var = create_new_tmp_var(block, dtype)
        block.append_op(
            type="fill_constant",
            outputs={'Out': [var]},
            attrs={
                'dtype': var.dtype,
                'shape': shape,
                'value': value,
                'force_cpu': False,
            },
            stop_gradient=True,
        )
        var.stop_gradient = True
        return var

    def create_scalar(block, value, dtype):
        return create_tensor(block, value, dtype, shape=[])

    def create_tensor_with_batchsize(ref_var, value, dtype):
        assert isinstance(ref_var, Variable)
        value = float(value)
        block = current_block(ref_var)
        var = create_new_tmp_var(block, dtype)
        batch_dim = -1
        out_shape = []
        for i, d in enumerate(ref_var.shape):
            if d < 0:
                if batch_dim < 0:
                    batch_dim = i
                    out_shape.append(d)
                else:
                    out_shape.append(1)
            else:
                out_shape.append(d)
        assert batch_dim != -1
        block.append_op(
            type='fill_constant_batch_size_like',
            outputs={'Out': [var]},
            inputs={'Input': [ref_var]},
            attrs={
                'shape': out_shape,
                'value': value,
                'input_dim_idx': batch_dim,
                'output_dim_idx': batch_dim,
            },
            stop_gradient=True,
        )

        var.stop_gradient = True
        return var

    @static_only
    def cpu(self):
        """
        In dy2static, Variable also needs cpu() and cuda() interface.
        But, the underneath operator has only forward op but not backward one.

        Returns:
            The tensor which has copied to cpu place.

        Examples:
            In Static Graph Mode:

            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()

                >>> x = paddle.static.data(name="x", shape=[2,2], dtype='float32')
                >>> y = x.cpu()
        """
        block = current_block(self)
        tmp_name = unique_tmp_name()
        output = block.create_var(
            name=tmp_name,
            dtype=self.dtype,
            shape=self.shape,
            type=self.type,
            persistable=False,
            stop_gradient=True,
        )
        # 0 means cpu place, see paddle/phi/kernels/memcpy_kernel.cc
        attrs = {'dst_place_type': 0}
        block.append_op(
            type='memcpy',
            inputs={'X': [self]},
            outputs={'Out': [output]},
            attrs=attrs,
        )
        return output

    @static_only
    def cuda(self, device_id=None, blocking=True):
        """
        In dy2static, Variable also needs cpu() and cuda() interface.
        But, the underneath operator has only forward op but not backward one.

        Args:
            self(Variable): The variable itself.
            device_id(int, optional): The destination GPU device id. Default: None, means current device.
                We add this argument for dy2static translation, please do not use it.
            blocking(bool, optional): Whether blocking or not, Default: True.
                We add this argument for dy2static translation, please do not use it.

        Returns:
            The tensor which has copied to cuda place.

        Examples:
            In Static Graph Mode:

            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()

                >>> x = paddle.static.data(name="x", shape=[2,2], dtype='float32')
                >>> y = x.cpu()
                >>> z = y.cuda()
        """
        if device_id is not None:
            warnings.warn("device_id is not supported, and it will be ignored.")
        if blocking is not True:
            warnings.warn("blocking is not supported, and it will be ignored.")

        block = current_block(self)
        tmp_name = unique_tmp_name()
        output = block.create_var(
            name=tmp_name,
            dtype=self.dtype,
            shape=self.shape,
            type=self.type,
            persistable=False,
            stop_gradient=True,
        )
        # 1 means cuda place, see paddle/phi/kernels/memcpy_kernel.cc
        attrs = {'dst_place_type': 1}
        block.append_op(
            type='memcpy',
            inputs={'X': [self]},
            outputs={'Out': [output]},
            attrs=attrs,
        )
        return output

    @static_only
    def place(self):
        """
        Variable don't have 'place' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a warning here and return None.
        """
        warnings.warn(
            "Variable do not have 'place' interface for static graph mode, try not to use it. None will be returned."
        )

    @static_only
    def contiguous(self):
        """
        Variable don't have 'contiguous' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a warning here and return None.
        """
        warnings.warn(
            "Variable do not have 'contiguous' interface for static graph mode, try not to use it. self will be returned."
        )
        return self

    @static_only
    def is_contiguous(self):
        """
        Variable don't have 'is_contiguous' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a warning here and return None.
        """
        warnings.warn(
            "Variable do not have 'is_contiguous' interface for static graph mode, try not to use it. True will be returned."
        )
        return True

    def astype(self, dtype):
        """
        **Notes**:
            **The variable must be a** :ref:`api_paddle_Tensor`

        Cast a variable to a specified data type if it differs from the current dtype;
        otherwise, return the original variable.

        Args:

            self(Variable): The source variable

            dtype: The target data type

        Returns:
            Variable: Variable with new dtype

        Examples:
            In Static Graph Mode:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.base as base
                >>> paddle.enable_static()
                >>> startup_prog = paddle.static.Program()
                >>> main_prog = paddle.static.Program()
                >>> with base.program_guard(startup_prog, main_prog):
                ...     original_variable = paddle.static.data(name = "new_variable", shape=[2,2], dtype='float32')
                ...     new_variable = original_variable.astype('int64')
                ...     print("new var's dtype is: {}".format(new_variable.dtype))
                ...
                new var's dtype is: paddle.int64

            In Dygraph Mode:

            .. code-block:: python

                >>> import paddle.base as base
                >>> import paddle
                >>> import numpy as np

                >>> x = np.ones([2, 2], np.float32) # type: ignore[var-annotated]
                >>> with base.dygraph.guard():
                ...     original_variable = paddle.to_tensor(x)
                ...     print("original var's dtype is: {}, numpy dtype is {}".format(original_variable.dtype, original_variable.numpy().dtype))
                ...     new_variable = original_variable.astype('int64')
                ...     print("new var's dtype is: {}, numpy dtype is {}".format(new_variable.dtype, new_variable.numpy().dtype))
                ...
                original var's dtype is: paddle.float32, numpy dtype is float32
                new var's dtype is: paddle.int64, numpy dtype is int64
        """
        if self.dtype == dtype:
            return self

        block = current_block(self)
        out = create_new_tmp_var(block, dtype)
        block.append_op(
            type="cast",
            inputs={"X": [self]},
            outputs={"Out": [out]},
            attrs={"in_dtype": self.dtype, "out_dtype": out.dtype},
        )
        out.stop_gradient = self.stop_gradient
        return out

    @static_only
    def append(self, var):
        """

        Note:
           The type variable must be LoD Tensor Array.

        """
        if not isinstance(var, Variable):
            if in_to_static_mode():
                """In dy2static mode, x may be tensor values such as int, float, np.array"""
                from paddle.tensor.creation import to_tensor

                var = to_tensor(var)
            else:
                raise TypeError(
                    f"Required input var should be Variable, but received {type(var)}"
                )
        if self.type != core.VarDesc.VarType.DENSE_TENSOR_ARRAY:
            raise TypeError(
                f"Only Variable with VarType.DENSE_TENSOR_ARRAY support `append` method, but received type: {self.type}"
            )
        from paddle.tensor.array import array_length, array_write

        array_write(x=var, i=array_length(self), array=self)

    @static_only
    def _item(self):
        """
        In order to be compatible with the item interface introduced by the dynamic graph, it does nothing but returns self.
        It will check that the shape must be a 1-D tensor
        """
        if len(self.shape) > 1:
            raise TypeError(
                f"Required input var should be 1-D Variable, but received {self.shape}"
            )
        return self

    @static_only
    def pop(self, *args):
        """
        The type variable must be LoD Tensor Array.
        When self is DenseTensorArray, calling pop is similar to Python's pop on list.
        This interface is used to simplify dygraph to static graph operations.

        Args:
            self(Variable): The source variable, which must be DENSE_TENSOR_ARRAY
            *args: optional, a int means index.
        Returns:
            Variable: self[index]
        """
        import paddle
        from paddle.static.nn import while_loop
        from paddle.tensor import fill_constant

        if self.type != core.VarDesc.VarType.DENSE_TENSOR_ARRAY:
            raise TypeError(
                f"Only Variable with VarType.DENSE_TENSOR_ARRAY support `pop` method, but received type: {self.type}"
            )
        if len(args) == 0:
            idx = -1
        else:
            idx = args[0]

        assert isinstance(idx, int)

        def cond(i, new_array):
            return paddle.less_than(i, arr_len)

        def body(i, new_array):
            item = paddle.tensor.array_read(array=self, i=i)
            paddle.tensor.array_write(
                item, paddle.tensor.array_length(new_array), new_array
            )

            i = paddle.increment(i)
            return i, new_array

        arr_len = paddle.tensor.array_length(self)
        if idx < 0:
            idx = idx + arr_len
        else:
            idx = fill_constant(shape=[1], dtype="int64", value=idx)

        pop_item = paddle.tensor.array_read(self, idx)

        tmp = paddle.assign(self)
        new_array = _slice_tensor_array(tmp, 0, idx)
        i = idx + 1

        _, new_array = while_loop(cond, body, [i, new_array])
        paddle.assign(new_array, output=self)

        return pop_item

    def _scalar_op_(var, scale, bias):
        block = current_block(var)
        out = create_new_tmp_var(block, var.dtype)
        block.append_op(
            type="scale",
            inputs={"X": [var]},
            outputs={"Out": [out]},
            attrs={"scale": scale, "bias": bias},
        )
        return out

    def _neg_(var):
        return _scalar_op_(var, -1.0, 0.0)

    def _abs_(var):
        return paddle.abs(var)

    @property
    def _ndim(self):
        """
        Returns the dimension of current Variable

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static Variable
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the Variable
                >>> print(x.ndim)
                3
        """
        return len(self.shape)

    def ndimension(self):
        """
        Returns the dimension of current Variable

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static Variable
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the Variable
                >>> print(x.ndimension())
                3
        """
        return len(self.shape)

    def dim(self):
        """
        Returns the dimension of current Variable

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static Variable
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the Variable
                >>> print(x.dim())
                3
        """
        return len(self.shape)

    def _scalar_add_(var, value):
        return _scalar_op_(var, 1.0, value)

    def _scalar_sub_(var, value):
        return _scalar_op_(var, 1.0, -value)

    def _scalar_rsub_(var, value):
        return _scalar_op_(var, -1.0, value)

    def _scalar_mul_(var, value):
        return _scalar_op_(var, value, 0.0)

    def _scalar_div_(var, value):
        return _scalar_op_(var, 1.0 / value, 0.0)

    def _binary_creator_(
        method_name, op_type, reverse=False, scalar_method=None
    ):
        def __impl__(self, other_var):
            # 1. scalar exists cases
            # we need combine the tensor.dtype and scalar.dtype, cast correct object
            if isinstance(other_var, float):
                # in all cases(+, -, *, /, **, //, %), we need cast tensor.dtype to float
                if self.dtype in _supported_int_dtype_:
                    self = astype(self, 'float32')
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            elif isinstance(other_var, int):
                # in all cases(+, -, *, /, **, //, %), we can cast it to float
                # because the output tensor.dtype depend on the type of input tensor
                other_var = float(other_var)
                # division is a special case
                # NOTE(chenweihang): because we cast tensor to float32 instead float64,
                # the division result can only guarantee the numerical accuracy of 6 digits
                # after the decimal point. The result of numpy calculation is of float64 type,
                # so the calculation result here and the calculation result of numpy are
                # different after 6 decimal point. If necessary, we can also use float64 here.
                # torch's behavior here is consistent with ours
                if (
                    op_type == 'elementwise_div'
                    and self.dtype in _supported_int_dtype_
                ):
                    self = astype(self, 'float32')
                # bool(tensor) + int(scalar) will do type promotion to int64
                if self.dtype == core.VarDesc.VarType.BOOL:
                    self = astype(self, 'int64')
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            elif isinstance(other_var, complex):
                if self.dtype not in _supported_complex_dtype_:
                    self = astype(self, 'complex64')
                    other_var = create_new_tmp_var(
                        current_block(self), dtype='complex64'
                    )
            else:
                # do nothing
                pass

            # 2. create variable for scalar
            lhs_dtype = safe_get_dtype(self)
            if not isinstance(other_var, Variable):
                if reverse:
                    for elem in self.shape:
                        if elem < 0:
                            other_var = create_tensor_with_batchsize(
                                self, other_var, lhs_dtype
                            )
                            break
                    else:
                        # when break is not triggered, enter the else branch
                        other_var = create_tensor(
                            current_block(self),
                            other_var,
                            dtype=lhs_dtype,
                            shape=self.shape,
                        )
                else:
                    # add fill_op to current_block
                    other_var = create_scalar(
                        current_block(self), value=other_var, dtype=lhs_dtype
                    )

            # 3. type promotion
            rhs_dtype = safe_get_dtype(other_var)

            if lhs_dtype != rhs_dtype:
                if method_name in SUPPORT_PROMOTION_OPS:
                    # different major types or both 0-d tensor follow with T+T rule.
                    if len(other_var.shape) == 0 or len(self.shape) == 0:
                        if not core.is_common_dtype_for_scalar(
                            lhs_dtype, rhs_dtype
                        ) or (
                            len(other_var.shape) == 0 and len(self.shape) == 0
                        ):
                            promote_type = core.get_promote_dtype_old_ir(
                                op_type, lhs_dtype, rhs_dtype
                            )
                            if lhs_dtype != promote_type:
                                self = astype(self, promote_type)
                            if rhs_dtype != promote_type:
                                other_var = astype(other_var, promote_type)
                        # common major types follow with tensor: int32(tensor) + int64(scalar) = int32
                        else:
                            if len(self.shape) == 0:
                                self = astype(self, rhs_dtype)
                            else:
                                other_var = astype(other_var, lhs_dtype)
                    elif core.need_type_promotion_old_ir(
                        op_type, lhs_dtype, rhs_dtype
                    ):
                        # only report warning here, real promotion deal in Executor
                        warnings.warn(
                            f"The input dtypes of OP {op_type} are {lhs_dtype} and {rhs_dtype}, the output will be auto-promoted"
                        )
                        warnings.filterwarnings(
                            "ignore", message="The input dtypes of OP"
                        )
                else:
                    raise TypeError(
                        f"got different data type in {op_type} between {lhs_dtype} and {rhs_dtype}."
                    )

            if reverse:
                tmp = self
                self = other_var
                other_var = tmp

            if (
                (op_type == "divide" or op_type == "elementwise_div")
                and self.dtype in _supported_int_dtype_
                and self.dtype == other_var.dtype
            ):
                self = astype(self, 'float32')
                other_var = astype(other_var, 'float32')
            # NOTE(zhiqiu): the output of compare operator should be bool.
            if method_name in compare_ops:
                out = create_new_tmp_var(current_block(self), dtype="bool")
            else:
                out = create_new_tmp_var(
                    current_block(self), dtype=safe_get_dtype(self)
                )

            axis = -1
            if other_var.ndim > 0 and other_var.shape[0] == -1:
                stack = inspect.stack()[1]
                file_name = stack[1]
                line_num = stack[2]
                warnings.warn(
                    f"{file_name}:{line_num}\nThe behavior of expression {EXPRESSION_MAP[method_name]} has been unified with {op_type}(X, Y, axis=-1) from Paddle 2.0. "
                    "If your code works well in the older versions but crashes in this version, try to use "
                    f"{op_type}(X, Y, axis=0) instead of {EXPRESSION_MAP[method_name]}. This transitional warning will be dropped in the future.",
                    category=DeprecationWarning,
                )
            current_block(self).append_op(
                type=op_type,
                inputs={'X': [self], 'Y': [other_var]},
                outputs={'Out': out},
                attrs={'axis': axis},
            )
            return out

        comment = OpProtoHolder.instance().get_op_proto(op_type).comment

        __impl__.__doc__ = f"""
        {comment}
        Args:
            self(Variable): left hand variable
            other_var(Variable|float|int): right hand variable

        Returns:
            Variable
        """
        __impl__.__name__ = method_name
        return __impl__

    def _int_(self):
        raise TypeError(
            "int(Variable) is not supported in static graph mode. If you are using @to_static, you can try this:\n"
            "1. If you want to get the value of Variable, you can switch to non-fullgraph mode by setting @to_static(full_graph=True).\n"
            "2. If you want to run it in full graph mode, you need use Variable.astype(paddle.int32), and do not use int(Variable)."
        )

    def _float_(self):
        raise TypeError(
            "float(Variable) is not supported in static graph mode. If you are using @to_static, you can try this:\n"
            "1. If you want to get the value of Variable, you can switch to non-fullgraph mode by setting @to_static(full_graph=True).\n"
            "2. If you want to run it in full graph mode, you need use Variable directly, and do not use float(Variable)."
        )

    def _complex_(self):
        raise TypeError(
            "complex(Variable) is not supported in static graph mode. If you are using @to_static, you can try this:\n"
            "1. If you want to get the value of Variable, you can switch to non-fullgraph mode by setting @to_static(full_graph=True).\n"
            "2. If you want to run it in full graph mode, you need use Variable directly, and do not use complex(Variable)."
        )

    def values(var):
        block = current_block(var)
        out = create_new_tmp_var(block, var.dtype)
        block.append_op(
            type="sparse_values",
            inputs={"x": [var]},
            outputs={"out": [out]},
            attrs={},
        )
        return out

    def indices(var):
        block = current_block(var)
        out = create_new_tmp_var(block, var.dtype)
        block.append_op(
            type="sparse_indices",
            inputs={"x": [var]},
            outputs={"out": [out]},
            attrs={},
        )
        return out

    def to_dense(var):
        block = current_block(var)
        out = create_new_tmp_var(block, var.dtype)
        block.append_op(
            type="sparse_to_dense",
            inputs={"x": [var]},
            outputs={"out": [out]},
            attrs={},
        )
        return out

    variable_methods = [
        #   b=-a
        ('__neg__', _neg_),
        ('__abs__', _abs_),
        ('astype', astype),
        ('cpu', cpu),
        ('cuda', cuda),
        ('place', place),
        ('contiguous', contiguous),
        ('is_contiguous', is_contiguous),
        ('append', append),
        ('item', _item),
        ('pop', pop),
        ('dim', dim),
        ('ndimension', ndimension),
        ('ndim', _ndim),
        (
            '__add__',
            _binary_creator_('__add__', 'elementwise_add', False, _scalar_add_),
        ),
        #  a+b == b+a. Do not need to reverse explicitly
        (
            '__radd__',
            _binary_creator_(
                '__radd__', 'elementwise_add', False, _scalar_add_
            ),
        ),
        (
            '__sub__',
            _binary_creator_('__sub__', 'elementwise_sub', False, _scalar_sub_),
        ),
        (
            '__rsub__',
            _binary_creator_(
                '__rsub__', 'elementwise_sub', True, _scalar_rsub_
            ),
        ),
        (
            '__mul__',
            _binary_creator_('__mul__', 'elementwise_mul', False, _scalar_mul_),
        ),
        #  a*b == b*a. Do not need to reverse explicitly
        (
            '__rmul__',
            _binary_creator_(
                '__rmul__', 'elementwise_mul', False, _scalar_mul_
            ),
        ),
        (
            '__div__',
            _binary_creator_('__div__', 'elementwise_div', False, _scalar_div_),
        ),
        (
            '__truediv__',
            _binary_creator_(
                '__truediv__', 'elementwise_div', False, _scalar_div_
            ),
        ),
        (
            '__rdiv__',
            _binary_creator_('__rdiv__', 'elementwise_div', True, None),
        ),
        (
            '__rtruediv__',
            _binary_creator_('__rtruediv__', 'elementwise_div', True, None),
        ),
        (
            '__pow__',
            _binary_creator_('__pow__', 'elementwise_pow', False, None),
        ),
        (
            '__rpow__',
            _binary_creator_('__rpow__', 'elementwise_pow', True, None),
        ),
        (
            '__floordiv__',
            _binary_creator_(
                '__floordiv__', 'elementwise_floordiv', False, None
            ),
        ),
        (
            '__rfloordiv__',
            _binary_creator_(
                '__rfloordiv__', 'elementwise_floordiv', True, None
            ),
        ),
        (
            '__mod__',
            _binary_creator_('__mod__', 'elementwise_mod', False, None),
        ),
        (
            '__rmod__',
            _binary_creator_('__rmod__', 'elementwise_mod', True, None),
        ),
        (
            '__matmul__',
            _binary_creator_('__matmul__', "matmul_v2", False, None),
        ),
        (
            '__rmatmul__',
            _binary_creator_('__rmatmul', "matmul_v2", True, None),
        ),
        #  for logical compare
        ('__eq__', _binary_creator_('__eq__', 'equal', False, None)),
        ('__ne__', _binary_creator_('__ne__', 'not_equal', False, None)),
        ('__lt__', _binary_creator_('__lt__', 'less_than', False, None)),
        ('__le__', _binary_creator_('__le__', 'less_equal', False, None)),
        ('__gt__', _binary_creator_('__gt__', 'greater_than', False, None)),
        ('__ge__', _binary_creator_('__ge__', 'greater_equal', False, None)),
        ('__float__', _float_),
        ('__int__', _int_),
        ('__complex__', _complex_),
        ('values', values),
        ('indices', indices),
        ('to_dense', to_dense),
    ]

    global _already_patch_variable
    if not _already_patch_variable:
        for method in variable_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(Variable, method_name, method_impl)
    else:
        import paddle.tensor

        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(Variable, method_name):
                continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl:
                setattr(Variable, method_name, method_impl)

        for magic_method, origin_method in paddle.tensor.magic_method_func:
            impl = getattr(paddle.tensor, origin_method, None)
            if impl:
                setattr(Variable, magic_method, impl)

    _already_patch_variable = True
