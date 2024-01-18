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

# TODO: define the extension functions


from paddle import _C_ops, tensor
from paddle.utils import deprecated

from ...base.data_feeder import check_type, check_variable_and_dtype
from ...base.layer_helper import LayerHelper
from ...common_ops_import import Variable
from ...framework import (
    convert_np_dtype_to_dtype_,
    core,
    in_dynamic_or_pir_mode,
)

__all__ = []


@deprecated(
    since="2.5.2",
    update_to="paddle.diag_embed",
    level=1,
    reason="diag_embed in paddle.nn.functional will be removed in future",
)
def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    return tensor.diag_embed(input, offset, dim1, dim2)


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

            >>> import paddle

            >>> lengths = paddle.to_tensor([10, 9, 8])
            >>> mask = paddle.nn.functional.sequence_mask(lengths)

            >>> print(mask)
            Tensor(shape=[3, 10], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

    """

    if in_dynamic_or_pir_mode():
        if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
            dtype = convert_np_dtype_to_dtype_(dtype)
        if maxlen is None:
            maxlen = -1
        out = _C_ops.sequence_mask(x, maxlen, dtype)
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

            >>> import paddle

            >>> ids = paddle.to_tensor([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]])

            >>> parents = paddle.to_tensor([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]])

            >>> final_sequences = paddle.nn.functional.gather_tree(ids, parents)
            >>> [[[2, 2], [1, 6]], [[3, 3], [6, 1]], [[0, 1], [9, 0]]]
            >>> final_sequences = paddle.nn.functional.gather_tree(ids, parents)
            >>> print(final_sequences)
            Tensor(shape=[3, 2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[2, 2],
              [1, 6]],
             [[3, 3],
              [6, 1]],
             [[0, 1],
              [9, 0]]])


    """
    if ids.ndim != 3:
        raise ValueError(
            "The input ids must be a 3D tensor with shape [length, batch_size, beam_size]"
        )
    if ids.ndim != parents.ndim:
        raise ValueError("The ids's shape must be the same as parents' shape. ")

    if in_dynamic_or_pir_mode():
        return _C_ops.gather_tree(ids, parents)
    else:
        helper = LayerHelper('gather_tree', **locals())
        check_variable_and_dtype(ids, 'ids', ['int32', 'int64'], 'gather_tree')
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


def temporal_shift(x, seg_num, shift_ratio=0.25, name=None, data_format="NCHW"):
    """

    **Temporal Shift Operator**

    Calculate the temporal shifting features for Input(X).

    Input(X) should be in shape of [N*T, C, H, W] or [N*T, H, W, C], while
    N is the batch size, T is the temporal segment number specified by
    :attr:`seg_num`, C is the channel number, H and W is the height and
    width of features.

    Temporal Shifting is calculated as follows when data format is NCHW:

    Step 1: Reshape Input(X) to [N, T, C, H, W].

    Step 2: Pad 0 to reshaping result in the 2nd(T) dimension with
    padding width as 1 on each side, padding result will be in shape
    of [N, T+2, C, H, W].

    Step 3: Assume :attr:`shift_ratio` is :math:`1/4`, slice padding
    result as follows:

    $$
    slice1 = x[:, :T, :C/4, :, :]
    $$
    $$
    slice2 = x[:, 2:T+2, C/4:C/2, :, :]
    $$
    $$
    slice3 = x[:, 1:T+1, C/2:, :, :]
    $$

    Step 4: Concatenate three slices along the 3rd(C) dimension and
    reshape result to [N*T, C, H, W].

    For details of temporal shifting, please refer to paper:
    `Temporal Shift Module <http://arxiv.org/abs/1811.08383>`_ .

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

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> input = paddle.randn([6, 4, 2, 2])
            >>> out = F.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)
    """
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. "
            f"Received Attr(data_format): {data_format}."
        )
    if in_dynamic_or_pir_mode():
        return _C_ops.temporal_shift(x, seg_num, shift_ratio, data_format)
    else:
        helper = LayerHelper("temporal_shift", **locals())
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64'],
            'temporal_shift',
        )
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
