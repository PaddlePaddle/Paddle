# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


import warnings

from paddle.base.libpaddle import DataType

from . import OpResult

_already_patch_opresult = False

_supported_int_dtype_ = [
    DataType.BOOL,
    DataType.UINT8,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
]


def create_tensor_with_batchsize(ref_var, value, dtype):
    assert isinstance(ref_var, OpResult)
    value = float(value)
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
    from paddle import _C_ops
    from paddle.framework import core

    out = _C_ops.full_batch_size_like(
        ref_var, out_shape, dtype, value, batch_dim, batch_dim, core.Place()
    )
    out.stop_gradient = True

    return out


def monkey_patch_opresult():
    def safe_get_dtype(var):
        try:
            dtype = var.dtype
        except:
            raise ValueError("Cannot get data type from var")
        return dtype

    def place(self):
        """
        OpResult don't have 'place' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a warnning here and return None.
        """
        warnings.warn(
            "OpResult do not have 'place' interface for pir graph mode, try not to use it. None will be returned."
        )

    @property
    def _ndim(self):
        """
        Returns the dimension of current OpResult

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static OpResult
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the OpResult
                >>> print(x.ndim)
                3
        """
        return len(self.shape)

    def ndimension(self):
        """
        Returns the dimension of current OpResult

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static OpResult
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the OpResult
                >>> print(x.ndimension())
                3
        """
        return len(self.shape)

    def dim(self):
        """
        Returns the dimension of current OpResult

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static OpResult
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the OpResult
                >>> print(x.dim())
                3
        """
        return len(self.shape)

    def _item(self):
        """
        In order to be compatible with the item interface introduced by the dynamic graph, it does nothing but returns self.
        It will check that the shape must be a 1-D tensor
        """
        if len(self.shape) > 1:
            raise TypeError(
                f"Required input var should be 1-D OpResult, but received {self.shape}"
            )
        return self

    def astype(self, dtype):
        """
        **Notes**:

        Cast a OpResult to a specified data type.

        Args:

            self(OpResult): The source OpResult

            dtype: The target data type

        Returns:
            OpResult: OpResult with new dtype

        Examples:
            In Static Graph Mode:

            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> startup_prog = paddle.static.Program()
                >>> main_prog = paddle.static.Program()
                >>> with paddle.static.program_guard(startup_prog, main_prog):
                ...     original_value = paddle.static.data(name = "new_value", shape=[2,2], dtype='float32')
                ...     new_value = original_value.astype('int64')
                ...     print("new value's dtype is: {}".format(new_value.dtype))
                ...
                new OpResult's dtype is: paddle.int64

        """
        from paddle import _C_ops

        if not isinstance(dtype, DataType):
            dtype = paddle.pir.core.convert_np_dtype_to_dtype_(dtype)
        return _C_ops.cast(self, dtype)

    def _scalar_add_(var, value):
        return paddle.scale(var, 1.0, value)

    def _scalar_sub_(var, value):
        return paddle.scale(var, 1.0, -value)

    def _scalar_rsub_(var, value):
        return paddle.scale(var, -1.0, value)

    def _scalar_mul_(var, value):
        return paddle.scale(var, value, 0.0)

    def _scalar_div_(var, value):
        return paddle.scale(var, 1.0 / value, 0.0)

    def _binary_creator_(
        method_name,
        python_api,
        reverse=False,
        scalar_method=None,
    ):
        def __impl__(self, other_var):
            # 1. scalar exists cases
            # we need combine the tensor.dtype and scalar.dtype, cast correct object
            if isinstance(other_var, float):
                # in all cases(+, -, *, /, **, //, %), we need cast tensor.dtype to float
                if self.dtype in _supported_int_dtype_:
                    self = astype(self, DataType.FLOAT32)
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
                    python_api == paddle.divide
                    and self.dtype in _supported_int_dtype_
                ):
                    paddle.cast(self, DataType.FLOAT32)
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            else:
                # do nothing
                pass

            # 2. create OpResult for scalar
            lhs_dtype = safe_get_dtype(self)
            other_var_opresult = other_var
            if not isinstance(other_var, OpResult):
                if reverse:
                    for elem in self.shape:
                        if elem < 0:
                            other_var_opresult = create_tensor_with_batchsize(
                                self, other_var, lhs_dtype
                            )

                            break
                    else:
                        # when break is not triggered, enter the else branch
                        other_var_opresult = paddle.fill_constant(
                            self.shape,
                            lhs_dtype,
                            other_var,
                        )
                else:
                    # add fill_op to current_block
                    other_var_opresult = paddle.fill_constant(
                        [],
                        lhs_dtype,
                        other_var,
                    )

            # 3. unify right var type to left var
            rhs_dtype = safe_get_dtype(other_var_opresult)
            if lhs_dtype != rhs_dtype:
                other_var_opresult = paddle.cast(other_var_opresult, lhs_dtype)
            if reverse:
                tmp = self
                self = other_var_opresult
                other_var_opresult = tmp

            if (
                python_api == paddle.divide
            ) and self.dtype in _supported_int_dtype_:
                self = paddle.cast(self, DataType.FLOAT32)
                other_var = paddle.cast(other_var_opresult, DataType.FLOAT32)

            out = python_api(self, other_var_opresult)
            return out

        __impl__.__doc__ = """
            Args:
                self(OpResult): left hand OpResult
                other_var(OpResult|float|int): right hand OpResult

            Returns:
                OpResult
            """
        __impl__.__name__ = method_name
        return __impl__

    import paddle

    opresult_methods = [
        ('place', place),
        ('item', _item),
        ('dim', dim),
        ('ndimension', ndimension),
        ('ndim', _ndim),
        ('astype', astype),
        (
            '__add__',
            _binary_creator_('__add__', paddle.tensor.add, False, _scalar_add_),
        ),
        #  a+b == b+a. Do not need to reverse explicitly
        (
            '__radd__',
            _binary_creator_(
                '__radd__', paddle.tensor.add, False, _scalar_add_
            ),
        ),
        (
            '__sub__',
            _binary_creator_(
                '__sub__', paddle.tensor.subtract, False, _scalar_sub_
            ),
        ),
        (
            '__rsub__',
            _binary_creator_(
                '__rsub__', paddle.tensor.subtract, True, _scalar_rsub_
            ),
        ),
        (
            '__mul__',
            _binary_creator_(
                '__mul__', paddle.tensor.multiply, False, _scalar_mul_
            ),
        ),
        #  a*b == b*a. Do not need to reverse explicitly
        (
            '__rmul__',
            _binary_creator_(
                '__rmul__', paddle.tensor.multiply, False, _scalar_mul_
            ),
        ),
        (
            '__div__',
            _binary_creator_(
                '__div__', paddle.tensor.divide, False, _scalar_div_
            ),
        ),
        (
            '__truediv__',
            _binary_creator_(
                '__truediv__', paddle.tensor.divide, False, _scalar_div_
            ),
        ),
        (
            '__rdiv__',
            _binary_creator_('__rdiv__', paddle.tensor.divide, True, None),
        ),
        (
            '__rtruediv__',
            _binary_creator_('__rtruediv__', paddle.tensor.divide, True, None),
        ),
        (
            '__pow__',
            _binary_creator_('__pow__', paddle.tensor.pow, False, None),
        ),
        (
            '__rpow__',
            _binary_creator_('__rpow__', paddle.tensor.pow, True, None),
        ),
        (
            '__floordiv__',
            _binary_creator_(
                '__floordiv__', paddle.tensor.floor_divide, False, None
            ),
        ),
        (
            '__mod__',
            _binary_creator_('__mod__', paddle.tensor.remainder, False, None),
        ),
        (
            '__matmul__',
            _binary_creator_('__matmul__', paddle.tensor.matmul, False, None),
        ),
    ]

    global _already_patch_opresult
    if not _already_patch_opresult:
        for method in opresult_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(OpResult, method_name, method_impl)

        # Handling Tensor Methods
        import paddle.tensor

        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(OpResult, method_name):
                continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl:
                setattr(OpResult, method_name, method_impl)

        # Handling __getitem__
        from ..base.variable_index import _getitem_static

        OpResult.__getitem__ = _getitem_static

        _already_patch_opresult = True
