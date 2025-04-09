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


import inspect
import textwrap
import warnings

import numpy as np

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

        Convert a value to a specified data type if it differs from the current dtype;
        otherwise, return the original value.

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

        if self.dtype == dtype:
            return self

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

    def _scalar_abs_(var):
        return paddle.abs(var)

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
                # bool(tensor) + int(scalar) will do type promotion to int64
                if self.dtype == paddle.bool:
                    self = paddle.cast(self, DataType.INT64)
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            else:
                # do nothing
                pass

            # 2. create Value for scalar
            lhs_dtype = safe_get_dtype(self)
            if not isinstance(other_var, Value):
                if reverse:
                    for elem in self.shape:
                        if elem < 0:
                            other_var = create_tensor_with_batchsize(
                                self, other_var, lhs_dtype
                            )

                            break
                    else:
                        # when break is not triggered, enter the else branch
                        other_var = paddle.tensor.creation.fill_constant(
                            self.shape,
                            lhs_dtype,
                            other_var,
                        )
                else:
                    # add fill_op to current_block
                    other_var = paddle.tensor.creation.fill_constant(
                        [],
                        lhs_dtype,
                        other_var,
                    )

            if reverse:
                tmp = self
                self = other_var
                other_var = tmp

            out = python_api(self, other_var)
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

    @property
    def _mT_(self):
        """

        Permute current Value with its last two dimensions reversed.

        If `n` is the dimensions of `x` , `x.mT` is equivalent to `x.transpose([0, 1, ..., n-1, n-2])`.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()

                >>> x = paddle.ones(shape=[2, 3, 5])
                >>> x_mT = x.mT

                >>> exe = paddle.static.Executor()
                >>> x_mT_np = exe.run(paddle.static.default_main_program(), fetch_list=[x_mT])[0]
                >>> print(x_mT_np.shape)
                (2, 5, 3)

        """
        if len(self.shape) < 2:
            raise ValueError(
                f"Tensor.ndim({len(self.shape)}) is required to be greater than or equal to 2."
            )

        perm = list(range(len(self.shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]

        return _C_ops.transpose(self, perm)

    def _int_(self):
        error_msg = """\
            int(Tensor) is not supported in static graph mode. Because it's value is not available during the static mode.
            It's usually triggered by the logging implicitly, for example:
                >>> logging.info("The value of x is: {int(x)}")
                                                          ^ `x` is Tensor, `int(x)` triggers int(Tensor)

                There are two common workarounds available:
                If you are logging Tensor values, then consider logging only at dynamic graphs, for example:

                    Modify the following code
                    >>> logging.info("The value of x is: {int(x)}")
                    to
                    >>> if paddle.in_dynamic_mode():
                    ...     logging.info("The value of x is: {int(x)}")

                If you need to convert the Tensor type, for example:
                    Modify the following code
                    >>> x = int(x)
                    to
                    >>> x = x.astype("int64")
        """

        raise TypeError(textwrap.dedent(error_msg))

    def _float_(self):
        error_msg = """\
            float(Tensor) is not supported in static graph mode. Because it's value is not available during the static mode.
            It's usually triggered by the logging implicitly, for example:
                >>> logging.info("The value of x is: {float(x)}")
                                                            ^ `x` is Tensor, `float(x)` triggers float(Tensor)

                There are two common workarounds available:
                If you are logging Tensor values, then consider logging only at dynamic graphs, for example:

                    Modify the following code
                    >>> logging.info("The value of x is: {float(x)}")
                    to
                    >>> if paddle.in_dynamic_mode():
                    ...     logging.info("The value of x is: {float(x)}")

                If you need to convert the Tensor type, for example:
                    Modify the following code
                    >>> x = float(x)
                    to
                    >>> x = x.astype("float64")
        """
        raise TypeError(textwrap.dedent(error_msg))

    def _bool_(self):
        error_msg = """\
            bool(Tensor) is not supported in static graph mode. Because it's value is not available during the static mode.
            If you haven't call bool(Tensor) explicitly, it's usually triggered by the control flow implicitly, for example:
                >>> if x > 0:
                       ^ `x` is Tensor, `x` > 0 is also a Tensor, `if x > 0` triggers bool(Tensor)
                ...     y = y + 1

            There are two common workarounds available:
            If you are checking for Tensor values, then consider checking only at dynamic graphs, for example:

                Modify the following code
                >>> if x > 0:
                ...     raise ValueError("x should be positive")
                to
                >>> if paddle.in_dynamic_mode() and x < 0:
                >>>     raise ValueError("x should be positive")

            If you need to control the flow of execution based on the value of the Tensor, then you need to rewrite the code as a control flow, for example:

                Modify the following code
                >>> if x < y:
                ...     y = y + 1
                ... else:
                ...     y = y - 1
                to
                >>> pred = paddle.less_than(x=x, y=y, name=None)
                >>> y = paddle.static.nn.cond(pred, lambda: y + 1, lambda: y - 1)
                For more info, please refer to https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/nn/cond_cn.html
            """
        raise TypeError(textwrap.dedent(error_msg))

    def _complex_(self):
        error_msg = """\
            complex(Tensor) is not supported in static graph mode. Because it's value is not available during the static mode.
            It's usually triggered by the logging implicitly, for example:
                >>> logging.info("The value of x is: {complex(x)}")
                                                              ^ `x` is Tensor, `complex(x)` triggers complex(Tensor)

                There are two common workarounds available:
                If you are logging Tensor values, then consider logging only at dynamic graphs, for example:

                    Modify the following code
                    >>> logging.info("The value of x is: {complex(x)}")
                    to
                    >>> if paddle.in_dynamic_mode():
                    ...     logging.info("The value of x is: {complex(x)}")

                If you need to convert the Tensor type, for example:
                    Modify the following code
                    >>> x = complex(x)
                    to
                    >>> x = x.astype("complex64")
        """
        raise TypeError(textwrap.dedent(error_msg))

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

    def to_dense(self):
        return _C_ops.sparse_to_dense(self)

    def values(self):
        return _C_ops.sparse_values(self)

    def indices(self):
        return _C_ops.sparse_indices(self)

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

    def _to(
        self,
        device=None,
        dtype=None,
        blocking=None,
    ):
        if device is None and dtype is None and blocking is None:
            return self

        if device is not None:
            if isinstance(device, str):
                device = paddle.device._convert_to_place(device)
            elif isinstance(
                device,
                (
                    paddle.core.Place,
                    paddle.CPUPlace,
                    paddle.CUDAPlace,
                    paddle.CUDAPinnedPlace,
                    # paddle.XPUPlace, # no support
                    # paddle.CustomPlace, # no support
                ),
            ):
                pass
            else:
                raise ValueError(
                    "device value error, must be str, paddle.CPUPlace(), paddle.CUDAPlace(), paddle.CUDAPinnedPlace(), paddle.XPUPlace() or paddle.CustomPlace(), but the type of device is "
                    + type(device).__name__
                )

        if blocking is None:
            blocking = True
        else:
            assert isinstance(
                blocking, bool
            ), "blocking value error, must be the True, False or None"

        def transform(t, device, dtype, blocking):
            if dtype is None:
                dtype = t.dtype
            t_used = t

            # 1. cast Tensor to dtype
            if dtype != t_used.dtype:
                with paddle.base.framework._dygraph_place_guard(
                    place=t_used.place
                ):
                    t_casted = t_used.cast(dtype=dtype)
            else:
                t_casted = t_used

            # 2. Copy casted Tensor(in CPU or GPU) to device
            if isinstance(device, paddle.CUDAPlace):
                new_t = t_casted.cuda(blocking=blocking)
            elif isinstance(device, paddle.CUDAPinnedPlace):
                if blocking is not True:
                    warnings.warn(
                        "blocking is not supported, and it will be ignored."
                    )
                new_t = _C_ops.memcpy(self, 2)
            elif isinstance(device, paddle.CPUPlace):
                new_t = t_casted.cpu()
            else:
                new_t = t_casted

            return new_t

        return transform(self, device, dtype, blocking)

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion. A paddle.dtype and place
        are inferred from the arguments of ``self.to(*args, **kwargs)``.There are
        three ways to call `to`:

            1. to(dtype, blocking=True)
            2. to(device, dtype=None, blocking=True)
            3. to(other, blocking=True)

        Returns:
            Tensor: self

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> tensorx = paddle.to_tensor([1,2,3])
                >>> print(tensorx)
                Tensor(shape=[3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                    [1, 2, 3])

                >>> tensorx = tensorx.to("cpu")
                >>> print(tensorx.place)
                Place(cpu)

                >>> tensorx = tensorx.to("float32")
                >>> print(tensorx.dtype)
                paddle.float32

                >>> tensorx = tensorx.to("gpu", "int16")
                >>> print(tensorx)
                Tensor(shape=[3], dtype=int16, place=Place(gpu:0), stop_gradient=True,
                    [1, 2, 3])
                >>> tensor2 = paddle.to_tensor([4,5,6])
                >>> tensor2
                Tensor(shape=[3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                    [4, 5, 6])
                >>> tensor2 = tensor2.to(tensorx)
                >>> print(tensor2)
                Tensor(shape=[3], dtype=int16, place=Place(gpu:0), stop_gradient=True,
                    [4, 5, 6])
        """

        size_args = len(args)
        size_kwargs = len(kwargs)

        if size_args + size_kwargs > 3 or size_args + size_kwargs == 0:
            raise TypeError(
                "to() received too many arguments - expected one of:\n  \
                * (Union[str, paddle.CPUPlace(), paddle.CUDAPlace(), paddle.CUDAPinnedPlace(), paddle.XPUPlace(), paddle.CustomPlace()] \
                device, Union[str, paddle.dtype, numpy.dtype] dtype, bool blocking)\n \
                * (Union[str, paddle.dtype, numpy.dtype] dtype, bool blocking)\n \
                * (paddle.Tensor other, bool blocking) "
            )
        valid_keys = {"device", "dtype", "blocking", "other"}
        invalid_keys = set(kwargs.keys()) - valid_keys
        if len(invalid_keys) != 0:
            raise TypeError(
                "to() got an unexpected keyword argument "
                + next(iter(invalid_keys))
            )

        def dtype_first_sig(dtype, blocking=None): ...

        def device_first_sig(device, dtype=None, blocking=None): ...

        def tensor_like_first_sig(other, blocking=None): ...

        class _NoArg: ...

        def is_dtype(arg):
            valid_dtypes = [
                "bfloat16",
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "complex64",
                "complex128",
                "bool",
            ]
            return isinstance(arg, (paddle.dtype, np.dtype)) or (
                isinstance(arg, str) and arg.lower() in valid_dtypes
            )

        def is_device(arg):
            # in dy2static, arg can be None
            return arg is None or isinstance(arg, (paddle.core.Place, str))

        def is_tensor(arg):
            return isinstance(arg, paddle.pir.Value)

        def create_positional_arg_extractor(position: int):
            def extract_positional_arg(args, kwargs):
                if len(args) > position:
                    return args[position]
                return _NoArg()

            return extract_positional_arg

        def create_keyword_arg_extractor(key: str, position: int):
            def extract_keyword_arg(args, kwargs):
                if (
                    key in kwargs
                    and len(kwargs) > position
                    and list(kwargs.keys())[position] == key
                ):
                    return kwargs[key]
                return _NoArg()

            return extract_keyword_arg

        def chain_extractors(*extractors):
            def chain(args, kwargs):
                for extractor in extractors:
                    if not isinstance(arg := extractor(args, kwargs), _NoArg):
                        return arg
                return _NoArg()

            return chain

        def dispatch_to_signature(*args, **kwargs):
            # dict[signature, (extractor, condition)]
            signature_map = {
                dtype_first_sig: (
                    chain_extractors(
                        create_positional_arg_extractor(position=0),
                        create_keyword_arg_extractor(key="dtype", position=0),
                    ),
                    is_dtype,
                ),
                device_first_sig: (
                    chain_extractors(
                        create_positional_arg_extractor(position=0),
                        create_keyword_arg_extractor(key="device", position=0),
                    ),
                    is_device,
                ),
                tensor_like_first_sig: (
                    chain_extractors(
                        create_positional_arg_extractor(position=0),
                        create_keyword_arg_extractor(key="other", position=0),
                    ),
                    is_tensor,
                ),
            }

            for sig, (extractor, condition) in signature_map.items():
                if not isinstance(
                    arg := extractor(args, kwargs), _NoArg
                ) and condition(arg):
                    bound_args = inspect.signature(sig).bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    return bound_args.arguments
            raise ValueError("No matching signature found.")

        args = dispatch_to_signature(*args, **kwargs)
        other = args.get("other", None)
        if other is not None:
            args.pop("other")
            args["dtype"] = other.dtype
            # in dy2static, we need show warning for this case
            other.place  # noqa: B018

        return self._to(**args)

    @fake_interface_only
    def numpy(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        Returns a numpy array shows the value of current :ref:`api_guide_Variable_en`
        Returns:
            ndarray: The numpy value of current Variable.
        Returns type:
            ndarray: dtype is same as current Variable
        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.base as base
                >>> from paddle.nn import Linear
                >>> import numpy as np
                >>> data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                >>> with base.dygraph.guard():
                ...     linear = Linear(32, 64)
                ...     data_tensor = paddle.to_tensor(data)
                ...     x = linear(data_tensor)
                ...     print(x.numpy())
        """
        pass

    @fake_interface_only
    def tolist(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        Returns a Python list that contains the elements of current :ref:`api_guide_Variable_en`

        Returns:
            list: The Python list containing the elements of current Variable.

        Returns type:
            list: Elements have the same dtype as current Variable

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.base as base
                >>> import numpy as np
                >>> data = np.random.uniform(-1, 1, [2, 3]).astype('float32')
                >>> with base.dygraph.guard():
                ...     x = paddle.to_tensor(data)
                ...     print(x.tolist())  # Convert tensor to Python list
        """
        pass

    @fake_interface_only
    def register_hook(self, hook):
        """
        Value don't have 'register_hook' interface in static graph mode
        But this interface can greatly facilitate dy2static.
        So we give a error here.
        """
        pass

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
        ('mT', _mT_),
        ('clone', clone),
        ('clear_gradient', clear_gradient),
        ('append', append),
        ('pop', pop),
        ('set_shape', set_shape),
        ('__hash__', value_hash),
        ('to_dense', to_dense),
        ('indices', indices),
        ('values', values),
        ("_to", _to),
        ("to", to),
        ("tolist", tolist),
        ("numpy", numpy),
        ("register_hook", register_hook),
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
            '__rfloordiv__',
            _binary_creator_(
                '__rfloordiv__', paddle.tensor.floor_divide, True, None
            ),
        ),
        (
            '__mod__',
            _binary_creator_('__mod__', paddle.tensor.remainder, False, None),
        ),
        (
            '__rmod__',
            _binary_creator_('__rmod__', paddle.tensor.remainder, True, None),
        ),
        (
            '__matmul__',
            _binary_creator_('__matmul__', paddle.tensor.matmul, False, None),
        ),
        (
            '__rmatmul__',
            _binary_creator_('__rmatmul__', paddle.tensor.matmul, True, None),
        ),
        ('__neg__', _scalar_neg_),
        ('__abs__', _scalar_abs_),
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
        ('__complex__', _complex_),
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
