#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define the extention functions

import numpy as np

from paddle import _C_ops, _legacy_C_ops, in_dynamic_mode

from ...fluid.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
)
from ...fluid.framework import (
    _in_legacy_dygraph,
    _non_static_mode,
    in_dygraph_mode,
)
from ...fluid.layer_helper import LayerHelper
from ...framework import convert_np_dtype_to_dtype_, core
from ...static import Variable
from ...tensor.creation import assign
from ...tensor.layer_function_generator import templatedoc

__all__ = []


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    """
    Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2)
    are filled by ``input``. By default, a 2D plane formed by the last two dimensions
    of the returned tensor will be selected.

    The argument ``offset`` determines which diagonal is generated:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    Args:
        input(Tensor|numpy.ndarray): The input tensor. Must be at least 1-dimensional. The input data type should be float32, float64, int32, int64.
        offset(int, optional): Which diagonal to consider. Default: 0 (main diagonal).
        dim1(int, optional): The first dimension with respect to which to take diagonal. Default: -2.
        dim2(int, optional): The second dimension with respect to which to take diagonal. Default: -1.

    Returns:
        Tensor, the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            diag_embed_input = paddle.arange(6)

            diag_embed_output1 = F.diag_embed(diag_embed_input)
            print(diag_embed_output1)
            # Tensor(shape=[6, 6], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 0, 0, 0, 0, 0],
            #         [0, 1, 0, 0, 0, 0],
            #         [0, 0, 2, 0, 0, 0],
            #         [0, 0, 0, 3, 0, 0],
            #         [0, 0, 0, 0, 4, 0],
            #         [0, 0, 0, 0, 0, 5]])

            diag_embed_output2 = F.diag_embed(diag_embed_input, offset=-1, dim1=0,dim2=1 )
            print(diag_embed_output2)
            # Tensor(shape=[7, 7], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 0, 0, 0, 0, 0, 0],
            #         [0, 0, 0, 0, 0, 0, 0],
            #         [0, 1, 0, 0, 0, 0, 0],
            #         [0, 0, 2, 0, 0, 0, 0],
            #         [0, 0, 0, 3, 0, 0, 0],
            #         [0, 0, 0, 0, 4, 0, 0],
            #         [0, 0, 0, 0, 0, 5, 0]])

            diag_embed_input_2dim = paddle.reshape(diag_embed_input,[2,3])
            print(diag_embed_input_2dim)
            # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 1, 2],
            #         [3, 4, 5]])
            diag_embed_output3 = F.diag_embed(diag_embed_input_2dim,offset= 0, dim1=0, dim2=2 )
            print(diag_embed_output3)
            # Tensor(shape=[3, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[[0, 0, 0],
            #          [3, 0, 0]],

            #         [[0, 1, 0],
            #          [0, 4, 0]],

            #         [[0, 0, 2],
            #          [0, 0, 5]]])
    """
    if not isinstance(input, Variable):
        input = assign(input)

    if in_dygraph_mode():
        return _C_ops.diag_embed(input, offset, dim1, dim2)
    elif in_dynamic_mode():
        return _legacy_C_ops.diag_embed(
            input, "offset", offset, "dim1", dim1, "dim2", dim2
        )

    inputs = {'Input': [input]}
    attrs = {'offset': offset, 'dim1': dim1, 'dim2': dim2}

    def __check_input(input, offset, dim1, dim2):
        check_dtype(
            input.dtype,
            'Input',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'diag_embed',
        )

        input_shape = list(input.shape)
        assert len(input_shape) >= 1, (
            "Input must be at least 1-dimensional, "
            "But received Input's dimensional: %s.\n" % len(input_shape)
        )

        assert np.abs(dim1) <= len(input_shape), (
            "Dim1 is out of range (expected to be in range of [%d, %d], but got %d).\n"
            % (-(len(input_shape) + 1), len(input_shape), dim1)
        )

        assert np.abs(dim2) <= len(input_shape), (
            "Dim2 is out of range (expected to be in range of [%d, %d], but got %d).\n"
            % (-(len(input_shape) + 1), len(input_shape), dim2)
        )

        dim1_ = dim1 if dim1 >= 0 else len(input_shape) + dim1 + 1
        dim2_ = dim2 if dim2 >= 0 else len(input_shape) + dim2 + 1
        assert dim1_ != dim2_, (
            "dim1 and dim2 cannot be the same dimension."
            "But received dim1 = %d, dim2 = %d\n" % (dim1, dim2)
        )

    __check_input(input, offset, dim1, dim2)
    helper = LayerHelper("diag_embed", **locals())

    out = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='diag_embed',
        inputs={'Input': [input]},
        attrs={'offset': offset, 'dim1': dim1, 'dim2': dim2},
        outputs={'Out': [out]},
    )
    out.stop_gradient = True
    return out


def sequence_mask(x, maxlen=None, dtype='int64', name=None):
    r"""
    **SequenceMask Layer**

    This layer outputs a mask according to the input :code:`x` and
    :code:`maxlen` with data type of :code:`dtype`.

    Supposing :code:`x` is a Tensor with shape [d_1, d_2, ..., d_n], the
    :code:`y` is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

    .. math::

        y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

    .. code-block:: text

        Case:

        Consider input:
            x = [3, 1, 1, 0]    max_len = 4

        then we get out:
            mask = [[1, 1, 1, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0]]

    Args:
        x (Variable): Input tensor of sequence_mask layer, \
            whose elements are integers less than :code:`maxlen`. \
            Tensor or LodTensor with shape [d_1, d_2, ..., d_n].
        maxlen (int, optional): Maximum length of the sequence. If :code:`maxlen` \
                           is None, it would be replace with :math:`max(x)`.
        dtype (np.dtype|paddle.dtype|str, optional): Data type of the output, \
             ``int64`` by default.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns:
            Tensor, The output sequence mask. Tensor with shape [d_1, d_2, ..., d_n, maxlen] \
            and data type of :code:`dtype`. The data type should be bool, float32, float64, int8, \
            int32 or int64.

    Examples:
        .. code-block:: python

            import paddle

            lengths = paddle.to_tensor([10, 9, 8])
            mask = paddle.nn.functional.sequence_mask(lengths)

            print(mask)
            # Tensor(shape=[3, 10], dtype=int64, place=Place(gpu:0), stop_gradient=True,
            #        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

    """

    if in_dygraph_mode():
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        if maxlen is not None:
            if isinstance(maxlen, core.eager.Tensor):
                attrs = ('out_dtype', dtype)
                out = _legacy_C_ops.sequence_mask(x, maxlen, *attrs)
            else:
                attrs = ('out_dtype', dtype, 'maxlen', maxlen)
                out = _legacy_C_ops.sequence_mask(x, None, *attrs)
            out.stop_gradient = True
            return out

    helper = LayerHelper('sequence_mask', **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)

    inputs = {'X': [x]}
    attrs = {'out_dtype': out.dtype}
    if maxlen is not None:
        if isinstance(maxlen, Variable):
            inputs['MaxLenTensor'] = maxlen
        else:
            attrs['maxlen'] = maxlen

    helper.append_op(
        type='sequence_mask', inputs=inputs, outputs={'Y': out}, attrs=attrs
    )

    out.stop_gradient = True
    return out


def gather_tree(ids, parents):
    r"""
    To be used after beam search. After beam search, we get selected ids at
    each time step and the corresponding parents in the search tree. Both ids
    and parents have the layout :attr:`[max_time, batch_size, beam_size]`. Then
    :attr:`gather_tree` is used to backtrace from the last time step and
    generate the full sequences by collecting selected ids.

    Here is an example:

    .. code-block:: text

            Given:
                ids = [[[2 2]
                        [6 1]]
                       [[3 9]
                        [6 1]]
                       [[0 1]
                        [9 0]]]
                parents = [[[0 0]
                            [1 1]]
                           [[1 0]
                            [1 0]]
                           [[0 0]
                            [0 1]]]

            Then:
                gather_tree(ids, parents)
                         = [[[2 2]
                             [1 6]]
                            [[3 3]
                             [6 1]]
                            [[0 1]
                             [9 0]]]

    Args:
        ids(Tensor): A Tensor with shape :attr:`[length, batch_size, beam_size]`
            and data type :attr:`int32` or :attr:`int64`. It contains the selected
            ids of all time steps.
        parents(Tensor): A Tensor with the same shape and data type as :attr:`ids`,
            It contains the parents corresponding to selected ids when searching
            among beams.

    Returns:
            A Tensor with the same shape and data type as :attr:`ids`. \
            It contains the full sequences. The sequences are collected from \
            :attr:`ids` by backtracing according to :attr:`parents`.

    Examples:
        .. code-block:: python

            import paddle

            ids = paddle.to_tensor([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]])

            parents = paddle.to_tensor([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]])

            final_sequences = paddle.nn.functional.gather_tree(ids, parents)
            # [[[2, 2], [1, 6]], [[3, 3], [6, 1]], [[0, 1], [9, 0]]]

    """
    if ids.ndim != 3:
        raise ValueError(
            "The input ids must be a 3D tensor with shape [length, batch_size, beam_size]"
        )
    if ids.ndim != parents.ndim:
        raise ValueError("The ids's shape must be the same as parents' shape. ")

    if in_dygraph_mode():
        return _C_ops.gather_tree(ids, parents)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.gather_tree(ids, parents)
        else:
            helper = LayerHelper('gather_tree', **locals())
            check_variable_and_dtype(
                ids, 'ids', ['int32', 'int64'], 'gather_tree'
            )
            check_variable_and_dtype(
                parents, 'parents', ['int32', 'int64'], 'gather_tree'
            )
            out = helper.create_variable_for_type_inference(dtype=ids.dtype)

            helper.append_op(
                type="gather_tree",
                inputs={"Ids": ids, "Parents": parents},
                outputs={"Out": out},
            )

            return out


@templatedoc()
def temporal_shift(x, seg_num, shift_ratio=0.25, name=None, data_format="NCHW"):
    """

    **Temporal Shift Operator**

    ${comment}

    Args:
        x(Tensor): ${x_comment}
        seg_num(int): ${seg_num_comment}
        shift_ratio(float): ${shift_ratio_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCHW" or "NHWC". Default: "NCHW".

    Returns:
        out(Tensor): The temporal shifting result is a tensor with the
        same shape and same data type as the input.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.randn([6, 4, 2, 2])
            out = F.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)
    """
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. "
            "Received Attr(data_format): {}.".format(data_format)
        )
    if in_dygraph_mode():
        return _C_ops.temporal_shift(x, seg_num, shift_ratio, data_format)
    if _non_static_mode():
        return _legacy_C_ops.temporal_shift(
            x,
            'seg_num',
            seg_num,
            'shift_ratio',
            shift_ratio,
            'data_format',
            data_format,
        )

    helper = LayerHelper("temporal_shift", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'temporal_shift')
    check_type(seg_num, 'seg_num', int, 'temporal_shift')
    check_type(shift_ratio, 'shift_ratio', float, 'temporal_shift')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(seg_num, int):
        raise TypeError("seg_num must be int type.")

    helper.append_op(
        type="temporal_shift",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={
            "seg_num": seg_num,
            "shift_ratio": shift_ratio,
            "data_format": data_format,
        },
    )
    return out
