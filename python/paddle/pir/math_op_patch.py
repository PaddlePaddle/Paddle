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

from paddle import _C_ops
from paddle.base.libpaddle import DataType
from paddle.base.wrapped_decorator import wrap_decorator

from . import Value

_already_patch_value = False

_supported_int_dtype_ = [
    DataType.BOOL,
    DataType.UINT8,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
]


def _fake_interface_only_(func):
    def __impl__(*args, **kwargs):
        raise AssertionError(
            f"'{func.__name__}' only can be called by `paddle.Tensor` in dynamic graph mode. Suggestions:\n"
            "  1. If you are in static graph mode, you can switch to dynamic graph mode by turning off `paddle.enable_static()` or calling `paddle.disable_static()`.\n"
            "  2. If you are using `@paddle.jit.to_static`, you can call `paddle.jit.enable_to_static(False)`. "
            f"If you have to translate dynamic graph to static graph, please use other API to replace '{func.__name__}'."
        )

    return __impl__


fake_interface_only = wrap_decorator(_fake_interface_only_)


def create_tensor_with_batchsize(ref_var, value, dtype):
    assert isinstance(ref_var, Value)
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

    from paddle.framework import core

    out = _C_ops.full_batch_size_like(
        ref_var, out_shape, dtype, value, batch_dim, batch_dim, core.Place()
    )
    out.stop_gradient = True

    return out


def monkey_patch_value():
    def safe_get_dtype(var):
        try:
            dtype = var.dtype
        except:
            raise ValueError("Cannot get data type from var")
        return dtype

    def cpu(self):
        """
        In dy2static, Value also needs cpu() and cuda() interface.
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

        # 0 means cpu place, see paddle/phi/kernels/memcpy_kernel.cc
        return _C_ops.memcpy(self, 0)

    def cuda(self, device_id=None, blocking=True):
        """
        In dy2static, Value also needs cpu() and cuda() interface.
        But, the underneath operator has only forward op but not backward one.

        Args:
            self(Value): The variable itself.
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

        # 1 means cuda place, see paddle/phi/kernels/memcpy_kernel.cc
        return _C_ops.memcpy(self, 1)

    @property
    def place(self):
        """
        Value don't have 'place' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a warning here and return None.
        """
        warnings.warn(
            "Value do not have 'place' interface for pir graph mode, try not to use it. None will be returned."
        )

    def contiguous(self):
        """
        Value don't have 'contiguous' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a warning here and return None.
        """
        warnings.warn(
            "Value do not have 'contiguous' interface for static graph mode, try not to use it. self will be returned."
        )
        return self

    def is_contiguous(self):
        """
        Value don't have 'is_contiguous' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a warning here and return None.
        """
        warnings.warn(
            "Value do not have 'is_contiguous' interface for static graph mode, try not to use it. True will be returned."
        )
        return True

    @property
    def _ndim(self):
        """
        Returns the dimension of current Value

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static Value
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the Value
                >>> print(x.ndim)
                3
        """
        return len(self.shape)

    def ndimension(self):
        """
        Returns the dimension of current Value

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static Value
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the Value
                >>> print(x.ndimension())
                3
        """
        return len(self.shape)

    def dim(self):
        """
        Returns the dimension of current Value

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static Value
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # print the dimension of the Value
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
                f"Required input var should be 1-D Value, but received {self.shape}"
            )
        return self

    def astype(self, dtype):
        """
        **Notes**:

        Cast a Value to a specified data type.

        Args:

            self(Value): The source Value

            dtype: The target data type

        Returns:
            Value: Value with new dtype

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
                ...     print(f"new value's dtype is: {new_value.dtype}")
                ...
                new Value's dtype is: paddle.int64

        """

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

    def _scalar_neg_(var):
        return paddle.scale(var, -1.0, 0.0)

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
                    self = paddle.cast(self, DataType.FLOAT32)
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            else:
                # do nothing
                pass

            # 2. create Value for scalar
            lhs_dtype = safe_get_dtype(self)
            other_var_value = other_var
            if not isinstance(other_var, Value):
                if reverse:
                    for elem in self.shape:
                        if elem < 0:
                            other_var_value = create_tensor_with_batchsize(
                                self, other_var, lhs_dtype
                            )

                            break
                    else:
                        # when break is not triggered, enter the else branch
                        other_var_value = paddle.tensor.creation.fill_constant(
                            self.shape,
                            lhs_dtype,
                            other_var,
                        )
                else:
                    # add fill_op to current_block
                    other_var_value = paddle.tensor.creation.fill_constant(
                        [],
                        lhs_dtype,
                        other_var,
                    )

            # 3. unify right var type to left var
            rhs_dtype = safe_get_dtype(other_var_value)
            if lhs_dtype != rhs_dtype:
                other_var_value = paddle.cast(other_var_value, lhs_dtype)
            if reverse:
                tmp = self
                self = other_var_value
                other_var_value = tmp

            if (
                python_api == paddle.divide
            ) and self.dtype in _supported_int_dtype_:
                self = paddle.cast(self, DataType.FLOAT32)
                other_var_value = paddle.cast(other_var_value, DataType.FLOAT32)

            out = python_api(self, other_var_value)
            return out

        __impl__.__doc__ = """
            Args:
                self(Value): left hand Value
                other_var(Value|float|int): right hand Value

            Returns:
                Value
            """
        __impl__.__name__ = method_name
        return __impl__

    @property
    def _size_(self):
        """
        Returns the number of elements for current Value, which is a int64 Value with shape [] .

        Returns:
            Value, the number of elements for current Value

        Examples:
            .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()
            >>> startup_prog = paddle.static.Program()
            >>> main_prog = paddle.static.Program()
            >>> with paddle.static.program_guard(startup_prog, main_prog):
            ...     x = paddle.assign(np.random.rand(2, 3, 4).astype("float32"))
            ...     (output_x,) = exe.run(main_program, fetch_list=[x.size])
            ...     print(f"value's size is: {output_x}")
            ...
            value's size is: 24
        """
        return paddle.numel(self)

    @property
    def _T_(self):
        """

        Permute current Value with its dimensions reversed.

        If `n` is the dimensions of `x` , `x.T` is equivalent to `x.transpose([n-1, n-2, ..., 0])`.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()

                >>> x = paddle.ones(shape=[2, 3, 5])
                >>> x_T = x.T

                >>> exe = paddle.static.Executor()
                >>> x_T_np = exe.run(paddle.static.default_main_program(), fetch_list=[x_T])[0]
                >>> print(x_T_np.shape)
                (5, 3, 2)

        """
        if len(self.shape) == 1:
            return self
        perm = list(reversed(range(len(self.shape))))

        return _C_ops.transpose(self, perm)

    def _int_(self):
        raise TypeError(
            "int(Value) is not supported in static graph mode. If you are using @to_static, you can try this:\n"
            "1. If you want to get the value of Value, you can switch to non-fullgraph mode by setting @to_static(full_graph=True).\n"
            "2. If you want to run it in full graph mode, you need use Value.astype(paddle.int32), and do not use int(Value)."
        )

    def _float_(self):
        raise TypeError(
            "float(Value) is not supported in static graph mode. If you are using @to_static, you can try this:\n"
            "1. If you want to get the value of Value, you can switch to non-fullgraph mode by setting @to_static(full_graph=True).\n"
            "2. If you want to run it in full graph mode, you need use Value directly, and do not use float(Value)."
        )

    def _bool_(self):
        raise TypeError(
            "bool(Value) is not supported in static graph mode. If you are using @to_static, you can try this:\n"
            "1. If you want to get the value of Value, you can switch to non-fullgraph mode by setting @to_static(full_graph=True).\n"
            "2. If you want to run it in full graph mode, you need use Value.astype(paddle.bool), and do not use bool(Value)."
        )

    def clone(self):
        """
        Returns a new static Value, which is the clone of the original static
        Value. It remains in the current graph, that is, the cloned Value
        provides gradient propagation. Calling ``out = tensor.clone()`` is same
        as ``out = assign(tensor)`` .

        Returns:
            Value, The cloned Value.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> # create a static Value
                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])
                >>> # create a cloned Value
                >>> y = x.clone()

        """
        return paddle.assign(self)

    @fake_interface_only
    def clear_gradient(self):
        """
        **Notes**:
            **1. This API is ONLY available in Dygraph mode**

            **2. Use it only Value has gradient, normally we use this for Parameters since other temporal Value will be deleted by Python's GC**

        Clear  (set to ``0`` ) the Gradient of Current Value

        Returns:  None

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import numpy as np

                >>> x = np.ones([2, 2], np.float32)
                >>> inputs2 = []
                >>> for _ in range(10):
                >>>     tmp = paddle.to_tensor(x)
                >>>     tmp.stop_gradient=False
                >>>     inputs2.append(tmp)
                >>> ret2 = paddle.add_n(inputs2)
                >>> loss2 = paddle.sum(ret2)
                >>> loss2.retain_grads()
                >>> loss2.backward()
                >>> print(loss2.gradient())
                >>> loss2.clear_gradient()
                >>> print("After clear {}".format(loss2.gradient()))
                1.0
                After clear 0.0
        """
        pass

    def append(self, var):
        """
        Notes:
           The type of Value must be Tensor Array.

        """
        if not self.is_dense_tensor_array_type():
            raise TypeError(
                f"Only Value with DenseTensorArray support `append` method, but received {self}"
            )
        from paddle.tensor.array import array_length, array_write

        array_write(x=var, i=array_length(self), array=self)

    def pop(self, *args):
        """
        The type of Value must be Tensor Array.
        When self is TensorArray, calling pop is similar to Python's pop on list.
        This interface is used to simplify dygraph to static graph operations.

        Args:
            self(Value): The source variable, which must be DenseTensorArray
            *args: optional, a int means index.
        Returns:
            Value: self[index]
        """

        if not self.is_dense_tensor_array_type():
            raise TypeError(
                f"Only Value with DenseTensorArray support `pop` method, but received {self}"
            )
        if len(args) == 0:
            idx = -1
        else:
            idx = args[0]

        return paddle._pir_ops.array_pop(self, idx)

    def set_shape(self, shape):
        assert (
            paddle.base.dygraph.base.in_to_static_mode()
        ), "We only support call 'set_shape' in to_static mode."

        if self.is_dense_tensor_type() or self.is_selected_row_type():
            type = paddle.pir.create_shaped_type(self.type(), shape)
            self.set_type(type)
        else:
            raise ValueError(
                "Currently, we can only set shape for dense and selected_row tensor"
            )

    def value_hash(self):
        return hash(id(self))

    import paddle

    value_methods = [
        ('cpu', cpu),
        ('cuda', cuda),
        ('place', place),
        ('contiguous', contiguous),
        ('is_contiguous', is_contiguous),
        ('item', _item),
        ('dim', dim),
        ('ndimension', ndimension),
        ('ndim', _ndim),
        ('astype', astype),
        ('size', _size_),
        ('T', _T_),
        ('clone', clone),
        ('clear_gradient', clear_gradient),
        ('append', append),
        ('pop', pop),
        ('set_shape', set_shape),
        ('__hash__', value_hash),
        # For basic operators
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
        ('__neg__', _scalar_neg_),
        # For compare operators
        (
            '__eq__',
            _binary_creator_('__eq__', paddle.tensor.equal, False, None),
        ),
        (
            '__ne__',
            _binary_creator_('__ne__', paddle.tensor.not_equal, False, None),
        ),
        (
            '__lt__',
            _binary_creator_('__lt__', paddle.tensor.less_than, False, None),
        ),
        (
            '__le__',
            _binary_creator_('__le__', paddle.tensor.less_equal, False, None),
        ),
        (
            '__gt__',
            _binary_creator_('__gt__', paddle.tensor.greater_than, False, None),
        ),
        (
            '__ge__',
            _binary_creator_(
                '__ge__', paddle.tensor.greater_equal, False, None
            ),
        ),
        ('__float__', _float_),
        ('__int__', _int_),
        ('__bool__', _bool_),
    ]

    global _already_patch_value
    if not _already_patch_value:
        for method in value_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(Value, method_name, method_impl)

        # Handling Tensor Methods
        import paddle.tensor

        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(Value, method_name):
                continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl:
                setattr(Value, method_name, method_impl)

        # Bit operation symbol
        for magic_method, origin_method in paddle.tensor.magic_method_func:
            impl = getattr(paddle.tensor, origin_method, None)
            if impl:
                setattr(Value, magic_method, impl)

        # Handling __getitem__
        from ..base.variable_index import _getitem_static, _setitem_static

        Value.__getitem__ = _getitem_static
        Value.__setitem__ = _setitem_static

        _already_patch_value = True
