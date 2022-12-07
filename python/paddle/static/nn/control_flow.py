#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


from paddle import _C_ops
from paddle.common_ops_import import LayerHelper
from paddle.fluid.data_feeder import check_type, check_variable_and_dtype
from paddle.framework import in_dygraph_mode

__all__ = [
    'Assert',
    'increment',
]


def Assert(cond, data=None, summarize=20, name=None):
    '''
    This API creates an op that asserts the given condition is true. If the
    condition is false, prints the tensors in data. ``summarize`` specifies the
    number of the elements in the tensors to print.

    Args:
        cond (Variable): The boolean condition tensor whose numel should be 1.
        data (list|tuple, optional): list or tuple of tensors to print when
            condition is not true. If it's ``None``, no tensor will be printed.
            The default value is ``None``.
        summarize (int, optional): Number of elements in the tensor to be
            printed. If its value is -1, then all elements in the tensor will
            be printed. The default value is 20.
        name (str, optional): The default value is ``None`` . Normally users
            don't have to set this parameter. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        Operator: the created operation.

    Raises:
        TypeError: If ``cond`` is not boolean Variable.
        TypeError: If ``data`` is not a list or tuple or ``None``.
        TypeError: If ``summarize`` is not int.
        TypeError: If ``name`` is not a string or ``None`` .
        framework.core.EnforceNotMet: If the condition is False in running time.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.static.nn.control_flow import Assert

            x = paddle.full([2, 3], 2.0, 'float32')
            condition = paddle.max(x) < 1.0 # False
            Assert(condition, [x], 10, "example_assert_layer")

            exe = paddle.static.Executor()
            try:
                exe.run(paddle.static.default_main_program())
                # Print x and throws paddle.framework.core.EnforceNotMet exception
                # Example printed message for x:
                #
                # Variable: fill_constant_0.tmp_0
                #   - lod: {}
                #   - place: CPUPlace()
                #   - shape: [2, 3]
                #   - layout: NCHW
                #   - dtype: float
                #   - data: [2 2 2 2 2 2]
            except paddle.framework.core.EnforceNotMet as e:
                print("Assert Exception Example")

    '''
    check_variable_and_dtype(
        cond, "cond", ["bool"], "static.nn.control_flow.Assert"
    )
    check_type(
        data, "data", (list, tuple, type(None)), "static.nn.control_flow.Assert"
    )
    check_type(summarize, "summarize", int, "static.nn.control_flow.Assert")
    check_type(name, "name", (str, type(None)), "static.nn.control_flow.Assert")

    layer_name = name if name else ('assert_' + cond.name)
    helper = LayerHelper(layer_name, **locals())

    op = helper.append_op(
        type="assert",
        inputs={"Cond": cond, "Data": [] if data is None else list(data)},
        attrs={"summarize": summarize},
    )

    return op


def increment(x, value=1.0, in_place=True):
    """
    The OP is usually used for control flow to increment the data of :attr:`x` by an amount :attr:`value`.
    Notice that the number of elements in :attr:`x` must be equal to 1.

    Parameters:
        x (Variable): A tensor that must always contain only one element, its data type supports
            float32, float64, int32 and int64.
        value (float, optional): The amount to increment the data of :attr:`x`. Default: 1.0.
        in_place (bool, optional): Whether the OP should be performed in-place. Default: True.

    Returns:
        Variable: The elementwise-incremented tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle
          from paddle.static.nn.control_flow import increment

          counter = paddle.zeros(shape=[1], dtype='float32') # [0.]
          increment(counter) # [1.]
    """
    if in_dygraph_mode():
        return _C_ops.increment_(x, value)

    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'increment'
    )
    helper = LayerHelper("increment", **locals())
    if not in_place:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = x
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': float(value)},
    )
    return out
