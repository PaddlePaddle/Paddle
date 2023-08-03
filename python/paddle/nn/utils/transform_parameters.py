# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from functools import reduce

import paddle
from paddle import _C_ops
from paddle.fluid.framework import (
    _create_tensor,
    _dygraph_tracer,
    dygraph_only,
    in_dygraph_mode,
)


# input==output, inplace strategy of reshape has no cost almostly
def _inplace_reshape_dygraph(x, shape):
    x_shape = _create_tensor(dtype='int64')
    if in_dygraph_mode():
        with paddle.fluid.dygraph.no_grad():
            tmp_out = _C_ops.reshape(x, shape)
            tmp_out._share_underline_tensor_to(x)
    else:
        _dygraph_tracer().trace_op(
            type="reshape2",
            inputs={'X': x},
            outputs={'Out': x, 'XShape': x_shape},
            attrs={'shape': shape},
            stop_gradient=True,
        )


@dygraph_only
def _stride_column(param):
    """
    A tool function. Permute date of parameter as a 'columns' stride. Now, it only support 2-D parameter.

    Args:
        param(Tensor]): The param that will be strided according to 'columns'.

    Examples:
       .. code-block:: python
            >>> # doctest: +SKIP('module paddle.nn.utils has no attribute stride_column')
            >>> import paddle
            >>> paddle.seed(100)

            >>> linear = paddle.nn.Linear(2, 3)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [[ 0.11732829, -0.64161885, -1.06996548],
                    [ 0.03456247, -0.29862350, -0.52380574]])

            >>> paddle.nn.utils.stride_column(linear.weight)
            >>> print(linear.weight)

    """
    assert len(param.shape) == 2
    shape = [param.shape[1], param.shape[0]]
    with paddle.fluid.dygraph.no_grad():
        reshape_var = paddle.reshape(param, shape)
        transpose_var = paddle.transpose(reshape_var, [1, 0])
        transpose_var._share_underline_tensor_to(param)


@dygraph_only
def parameters_to_vector(parameters, name=None):
    """
    Flatten parameters to a 1-D Tensor.

    Args:
        parameters(Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A 1-D Tensor, which represents the parameters of a Layer.


    Examples:
       .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> linear = paddle.nn.Linear(10, 15)

            >>> paddle.nn.utils.parameters_to_vector(linear.parameters())
            Tensor(shape=[165], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [ 0.35844472,  0.01974013, -0.23553374,  0.39706543, -0.07446361,
                    -0.09169629,  0.46070877,  0.23943624,  0.01749060,  0.22822258,
                     0.46821097, -0.44761088, -0.07211867,  0.21514717, -0.28598809,
                    -0.29656941, -0.11341774, -0.34681654, -0.26734930, -0.05283538,
                    -0.27769512,  0.17328075, -0.03468305,  0.41526547, -0.02535823,
                    -0.20390061,  0.28938183,  0.24982938, -0.27144754,  0.31550846,
                    -0.33570367,  0.34471479,  0.47482201, -0.00958112,  0.29664621,
                    -0.09316105,  0.37654552,  0.42216530,  0.43258956,  0.02038971,
                    -0.22948775,  0.31592992,  0.24399504, -0.20746091,  0.13301626,
                    -0.27582464, -0.06244987,  0.48669496, -0.29297251, -0.18692544,
                     0.45354423,  0.09079853,  0.40627977, -0.38706431, -0.24094580,
                     0.20904800, -0.36868426, -0.38044789,  0.10385343, -0.15778929,
                     0.37157646, -0.01948500,  0.40487817, -0.38128233,  0.38129857,
                     0.36317143,  0.43580243, -0.47675842,  0.36897883,  0.44193909,
                     0.03970906, -0.07966241, -0.28031254,  0.17790404, -0.45868829,
                    -0.05428463,  0.00965381,  0.21480617, -0.48450279,  0.36737546,
                    -0.42163223,  0.34161910,  0.28201506,  0.35220489, -0.01180339,
                    -0.11615443,  0.00337344, -0.20239002, -0.30462277,  0.40330753,
                    -0.43384531, -0.22517624,  0.17520681,  0.00863490, -0.43984908,
                     0.09856793,  0.01858523, -0.06763247, -0.38120067, -0.24793185,
                    -0.05841520,  0.38417909, -0.17650360, -0.48265788,  0.32629517,
                     0.08259115,  0.25128672,  0.11733320, -0.09207663,  0.20171347,
                     0.39848068,  0.38935456, -0.20116216,  0.35647616,  0.01518515,
                     0.44569293, -0.25974736, -0.07736969,  0.03493360, -0.26575688,
                    -0.02520940,  0.46487620,  0.42347303, -0.37944031,  0.24258563,
                    -0.15446970,  0.19769254,  0.03239343,  0.15246859, -0.38727424,
                     0.22904631, -0.01441202, -0.25648475, -0.29074892, -0.28782564,
                     0.13929120,  0.22800788,  0.31670925, -0.44742560, -0.14841846,
                    -0.12636486, -0.45744216, -0.33645833, -0.46459571, -0.27040026,
                     0.23058102, -0.36085564, -0.33098835, -0.41679865, -0.19043070,
                     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                     0.        ,  0.        ,  0.        ,  0.        ,  0.        ])

    """
    dtype = parameters[0].dtype
    origin_shapes = []
    for param in parameters:
        origin_shapes.append(param.shape)
        _inplace_reshape_dygraph(param, [-1])

    out = _create_tensor(dtype=dtype)
    if in_dygraph_mode():
        with paddle.fluid.dygraph.no_grad():
            tmp = _C_ops.concat(parameters, 0)
            tmp._share_underline_tensor_to(out)
    else:
        _dygraph_tracer().trace_op(
            type='concat',
            inputs={'X': parameters},
            outputs={'Out': [out]},
            attrs={'axis': 0},
            stop_gradient=True,
        )
    for i, param in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])
    return out


@dygraph_only
def vector_to_parameters(vec, parameters, name=None):
    """
    Transform a 1-D Tensor to the input ``parameters`` .

    Args:
        vec (Tensor): A 1-D Tensor, which will be sliced and copied to the input ``parameters`` .
        parameters (Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
       .. code-block:: python

            >>> import paddle
            >>> weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(3.))
            >>> linear1 = paddle.nn.Linear(10, 15, weight_attr)

            >>> vec = paddle.nn.utils.parameters_to_vector(linear1.parameters())

            >>> linear2 = paddle.nn.Linear(10, 15)
            # copy weight of linear1 to linear2
            >>> paddle.nn.utils.vector_to_parameters(vec, linear2.parameters())
            >>> print(linear2.weight)
            Parameter containing:
            Tensor(shape=[10, 15], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [[3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]])
    """
    origin_shapes = []
    sections = []
    for param in parameters:
        shape = param.shape
        origin_shapes.append(shape)
        numel = reduce(lambda x, y: x * y, shape, 1)
        sections.append(numel)

    if len(sections) == 1:
        sections.append(0)

    if in_dygraph_mode():
        with paddle.fluid.dygraph.no_grad():
            res = _C_ops.split(vec, sections, 0)
            for i in range(0, len(parameters)):
                res[i]._share_underline_tensor_to(parameters[i])
    else:
        _dygraph_tracer().trace_op(
            type='split',
            inputs={'X': [vec]},
            outputs={'Out': parameters},
            attrs={'axis': 0, 'sections': sections},
            stop_gradient=True,
        )

    for i, param in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])
    return
