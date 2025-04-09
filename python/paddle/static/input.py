# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

import paddle
from paddle.base import Variable, core
from paddle.base.data_feeder import check_type
from paddle.base.framework import (
    convert_np_dtype_to_dtype_,
    in_pir_mode,
    static_only,
)
from paddle.base.layer_helper import LayerHelper
from paddle.base.libpaddle import DataType
from paddle.base.libpaddle.pir import (
    get_current_insertion_point,
    set_insertion_point,
)

from ..base.variable_index import _setitem_static

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import (
        DTypeLike,
        ShapeLike,
        Size1,
        TensorIndex,
        TensorLike,
    )

__all__ = []


def evaluate_flag(val) -> bool:
    return str(val).lower() not in ('false', 'off', '0', 'none')


@static_only
def data(
    name: str,
    shape: ShapeLike,
    dtype: DTypeLike | None = None,
    lod_level: int = 0,
) -> paddle.Tensor:
    """

    This function creates a variable on the global block. The global variable
    can be accessed by all the following operators in the graph. The variable
    is a placeholder that could be fed with input, such as Executor can feed
    input into the variable. When `dtype` is None, the dtype
    will get from the global dtype by `paddle.get_default_dtype()`.

    Args:
        name (str): The name/alias of the variable, see :ref:`api_guide_Name`
            for more details.
        shape (list|tuple): List|Tuple of integers declaring the shape. You can
            set None or -1 at a dimension to indicate the dimension can be of any
            size. For example, it is useful to set changeable batch size as None or -1.
        dtype (np.dtype|str, optional): The type of the data. Supported
            dtype: bool, float16, float32, float64, int8, int16, int32, int64,
            uint8. Default: None. When `dtype` is not set, the dtype will get
            from the global dtype by `paddle.get_default_dtype()`.
        lod_level (int, optional): The LoD level of the DenseTensor. Usually users
            don't have to set this value. Default: 0.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP("This has diff in xdoctest env")
            >>> import numpy as np
            >>> import paddle
            >>> paddle.enable_static()

            # Creates a variable with fixed size [3, 2, 1]
            # User can only feed data of the same shape to x
            # the dtype is not set, so it will set "float32" by
            # paddle.get_default_dtype(). You can use paddle.get_default_dtype() to
            # change the global dtype
            >>> x = paddle.static.data(name='x', shape=[3, 2, 1])

            # Creates a variable with changeable batch size -1.
            # Users can feed data of any batch size into y,
            # but size of each data sample has to be [2, 1]
            >>> y = paddle.static.data(name='y', shape=[-1, 2, 1], dtype='float32')

            >>> z = x + y

            # In this example, we will feed x and y with np-ndarray "1"
            # and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
            >>> feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32) # type: ignore[var-annotated]

            >>> exe = paddle.static.Executor(paddle.framework.CPUPlace())
            >>> out = exe.run(paddle.static.default_main_program(),
            ...             feed={
            ...                 'x': feed_data,
            ...                 'y': feed_data
            ...             },
            ...             fetch_list=[z.name])

            # np-ndarray of shape=[3, 2, 1], dtype=float32, whose elements are 2
            >>> print(out)
            [array([[[2.],
                    [2.]],
                   [[2.],
                    [2.]],
                   [[2.],
                    [2.]]], dtype=float32)]

    """

    def _reset_data_op_insertion_point():
        default_main_program = paddle.pir.core.default_main_program()
        ops = default_main_program.global_block().ops
        if len(ops) == 0:
            return
        for op in ops:
            if op.name() != 'pd_op.data':
                paddle.pir.set_insertion_point(op)
                return

    helper = LayerHelper('data', **locals())
    check_type(name, 'name', (bytes, str), 'data')
    check_type(shape, 'shape', (list, tuple), 'data')

    shape = list(shape)
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = -1
        if isinstance(shape[i], int) and shape[i] < 0 and shape[i] != -1:
            raise ValueError(
                f"Only -1 can be used in shape to indicate unknown dimension, but received {shape[i]}"
            )

    if dtype is None:
        dtype = paddle.get_default_dtype()

    if in_pir_mode():
        ir_dtype = dtype
        if not isinstance(ir_dtype, DataType):
            ir_dtype = paddle.pir.core.convert_np_dtype_to_dtype_(dtype)
        prev_insertion_point = get_current_insertion_point()
        _reset_data_op_insertion_point()
        out = paddle._pir_ops.data(name, shape, ir_dtype, core.Place())
        set_insertion_point(prev_insertion_point)
        return out

    out = helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=core.VarDesc.VarType.DENSE_TENSOR,
        stop_gradient=True,
        lod_level=lod_level,
        is_data=True,
        need_check_feed=True,
    )

    is_pir_mode = os.environ.get("FLAGS_enable_pir_in_executor", None)
    if evaluate_flag(is_pir_mode):
        helper = LayerHelper('data', **locals())
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        helper.append_op(
            type='data',
            inputs={},
            outputs={'out': out},
            attrs={
                'shape': shape,
                'dtype': dtype,
                'place': 0,
                'name': name,
            },
        )
    return out


class InputSpec:
    """
    InputSpec describes the signature information of the model input, such as ``shape`` , ``dtype`` , ``name`` .

    This interface is often used to specify input tensor information of models in high-level API.
    It's also used to specify the tensor information for each input parameter of the forward function
    decorated by `@paddle.jit.to_static`.

    Args:
        shape (tuple(integers)|list[integers]): List|Tuple of integers
            declaring the shape. You can set "None" or -1 at a dimension
            to indicate the dimension can be of any size. For example,
            it is useful to set changeable batch size as "None" or -1.
        dtype (np.dtype|str, optional): The type of the data. Supported
            dtype: bool, float16, float32, float64, int8, int16, int32, int64,
            uint8. Default: float32.
        name (str): The name/alias of the variable, see :ref:`api_guide_Name`
            for more details.
        stop_gradient (bool, optional): A boolean that mentions whether gradient should flow. Default is False, means don't stop calculate gradients.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.static import InputSpec

            >>> input = InputSpec([None, 784], 'float32', 'x')
            >>> label = InputSpec([None, 1], 'int64', 'label')

            >>> print(input)
            InputSpec(shape=(-1, 784), dtype=paddle.float32, name=x, stop_gradient=False)

            >>> print(label)
            InputSpec(shape=(-1, 1), dtype=paddle.int64, name=label, stop_gradient=False)
    """

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = 'float32',
        name: str | None = None,
        stop_gradient: bool = False,
    ) -> None:
        # replace `None` in shape  with -1
        self.shape = self._verify(shape)
        # convert dtype into united representation
        if dtype is not None:
            if isinstance(dtype, (np.dtype, str)):
                dtype = convert_np_dtype_to_dtype_(dtype)

        self.dtype = dtype
        self.name = name
        self.stop_gradient = stop_gradient

    def _create_feed_layer(self):
        return data(self.name, shape=self.shape, dtype=self.dtype)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, name={self.name}, stop_gradient={self.stop_gradient})'

    @classmethod
    def from_tensor(cls, tensor: Tensor, name: str | None = None) -> Self:
        """
        Generates a InputSpec based on the description of input tensor.

        Args:
            tensor(Tensor): the source tensor to generate a InputSpec instance

        Returns:
            A InputSpec instance generated from Tensor.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.static import InputSpec

                >>> paddle.disable_static()

                >>> x = paddle.ones([2, 2], dtype="float32")
                >>> x_spec = InputSpec.from_tensor(x, name='x')
                >>> print(x_spec)
                InputSpec(shape=(2, 2), dtype=paddle.float32, name=x, stop_gradient=False)

        """
        if isinstance(tensor, (Variable, core.eager.Tensor, paddle.pir.Value)):
            return cls(tensor.shape, tensor.dtype, name or tensor.name)
        else:
            raise ValueError(
                f"Input `tensor` should be a Tensor, but received {type(tensor).__name__}."
            )

    @classmethod
    def from_numpy(
        cls, ndarray: npt.NDArray[Any], name: str | None = None
    ) -> Self:
        """
        Generates a InputSpec based on the description of input np.ndarray.

        Args:
            tensor(Tensor): the source numpy ndarray to generate a InputSpec instance

        Returns:
            A InputSpec instance generated from Tensor.

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> from paddle.static import InputSpec

                >>> x = np.ones([2, 2], np.float32) # type: ignore[var-annotated]
                >>> x_spec = InputSpec.from_numpy(x, name='x')
                >>> print(x_spec)
                InputSpec(shape=(2, 2), dtype=paddle.float32, name=x, stop_gradient=False)

        """
        return cls(ndarray.shape, ndarray.dtype, name)

    def batch(self, batch_size: int | Size1) -> Self:
        """
        Inserts `batch_size` in front of the `shape`.

        Args:
            batch_size(int): the inserted integer value of batch size.

        Returns:
            The original InputSpec instance by inserting `batch_size` in front of `shape`.

        Examples:
            .. code-block:: python

                >>> from paddle.static import InputSpec

                >>> x_spec = InputSpec(shape=[64], dtype='float32', name='x')
                >>> x_spec.batch(4)
                >>> print(x_spec)
                InputSpec(shape=(4, 64), dtype=paddle.float32, name=x, stop_gradient=False)

        """
        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) != 1:
                raise ValueError(
                    f"Length of batch_size: {batch_size} shall be 1, but received {len(batch_size)}."
                )
            batch_size = batch_size[0]
        elif not isinstance(batch_size, int):
            raise TypeError(
                f"type(batch_size) shall be `int`, but received {type(batch_size).__name__}."
            )

        new_shape = [batch_size, *list(self.shape)]
        self.shape = tuple(new_shape)

        return self

    def unbatch(self) -> Self:
        """
        Removes the first element of `shape`.

        Returns:
            The original InputSpec instance by removing the first element of `shape` .

        Examples:
            .. code-block:: python

                >>> from paddle.static import InputSpec

                >>> x_spec = InputSpec(shape=[4, 64], dtype='float32', name='x')
                >>> x_spec.unbatch()
                >>> print(x_spec) # InputSpec(shape=(64,), dtype=paddle.float32, name=x)
                InputSpec(shape=(64,), dtype=paddle.float32, name=x, stop_gradient=False)

        """
        if len(self.shape) == 0:
            raise ValueError(
                "Not support to unbatch a InputSpec when len(shape) == 0."
            )

        self.shape = self._verify(self.shape[1:])
        return self

    def _verify(self, shape):
        """
        Verifies the input shape and modifies `None` into `-1`.
        """
        if not isinstance(shape, (list, tuple)):
            raise TypeError(
                f"Type of `shape` in InputSpec should be one of (tuple, list), but received {type(shape).__name__}."
            )

        for i, ele in enumerate(shape):
            if ele is not None:
                if not isinstance(ele, int):
                    raise ValueError(
                        f"shape[{i}] should be an `int`, but received `{type(ele).__name__}`:{ele}."
                    )
            if ele is None or ele < -1:
                shape[i] = -1

        return tuple(shape)

    def __hash__(self) -> int:
        # Note(Aurelius84): `name` is not considered as a field to compute hashkey.
        # Because it's no need to generate a new program in following cases while using
        # @paddle.jit.to_static.
        #
        # Case 1:
        #      foo(x_var)
        #      foo(y_var)
        #  x_var and y_var hold same shape and dtype, they should share a same program.
        #
        #
        # Case 2:
        #      foo(x_var)
        #      foo(x_np)  # x_np is a numpy.ndarray.
        #  x_var and x_np hold same shape and dtype, they should also share a same program.
        return hash((tuple(self.shape), self.dtype, self.stop_gradient))

    def __eq__(self, other: Self) -> bool:
        slots = ['shape', 'dtype', 'name', 'stop_gradient']
        return type(self) is type(other) and all(
            getattr(self, attr) == getattr(other, attr) for attr in slots
        )

    def __ne__(self, other) -> bool:
        return not self == other


def setitem(
    x: Tensor,
    index: TensorIndex,
    value: TensorLike,
) -> Tensor:
    """
    x(Tensor): input Tensor.
    index(Scalar|Tuple|List|Tensor): Where should be set value.
    value(Scalar|Tensor): The value which is going to be set.

    [How to write index?]
    1. ':' -> slice(),
       (1) a[:]=v -> setitem(a, slice(None,None,None), v)
       (2) a[1::2] -> setitem(a, slice(1,None,2), v)

    2. if there are multiple indexes for axes, use TUPLE (Not LIST) to pack them.
       (1) a[1, 2]=v -> setitem(a, (1, 2), v)
       (2) a[[1,2],[2,3]]=v -> setitem(a, ([1,2],[2,3]), v)
       (3) a[1,:, 3] = v -> setitem(a, (1, slice(None,None,None),3), v)
       (4) a[1, ..., 2]=v -> setitem(a, (1, ..., 2), v)

    3. You can always use TUPLE as index input, even there is only one index.
       (1) a[Tensor([10,10])]=v -> setitem(a, (Tensor([10,10]),), v)
       (2) a[1] = v -> setitem(a, (1,), v)
    """
    return _setitem_static(x, index, value)
