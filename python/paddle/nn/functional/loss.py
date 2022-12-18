# -*- coding: utf-8 -*
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

# TODO: define loss functions of neural network
import paddle
import paddle.fluid as fluid
from paddle import _C_ops, _legacy_C_ops, in_dynamic_mode
from paddle.framework import core
from paddle.utils import deprecated

from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.framework import (
    _current_expected_place,
    _in_legacy_dygraph,
    _non_static_mode,
    _varbase_creator,
    in_dygraph_mode,
)
from ...fluid.layer_helper import LayerHelper
from ...fluid.layers.nn import _elementwise_op_in_dygraph
from ...static import Variable
from ...tensor.manipulation import reshape

__all__ = []

kIgnoreIndex = -100


def dice_loss(input, label, epsilon=0.00001, name=None):
    r"""

    Dice loss for comparing the similarity between the input predictions and the label.
    This implementation is for binary classification, where the input is sigmoid
    predictions of each pixel, usually used for segmentation task. The dice loss can
    be defined as the following equation:

    .. math::

        dice\_loss &= 1 - \frac{2 * intersection\_area}{total\_area} \\
                  &= \frac{(total\_area - intersection\_area) - intersection\_area}{total\_area} \\
                  &= \frac{(union\_area - intersection\_area)}{total\_area}


    Parameters:
        input (Tensor): Tensor, rank>=2, shape is :math:`[N_1, N_2, ..., N_k, D]`, where :math:`N_1` is
                          the batch_size, :math:`D` is the number of categories. It is usually the output
                          predictions of sigmoid activation. The data type can be float32 or float64.
        label (Tensor): Tensor, the groud truth with the same rank as input, shape is :math:`[N_1, N_2, ..., N_k, 1]`.
                          where :math:`N_1` is the batch_size. The data type can be int32 or int64.
        epsilon (float): The epsilon will be added to the numerator and denominator.
                         If both input and label are empty, it makes sure dice is 1.
                         Default: 0.00001
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor, which shape is [1], data type is the same as `input` .

    Example:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            x = paddle.randn((3,224,224,2))
            label = paddle.randint(high=2, shape=(3,224,224,1))
            predictions = F.softmax(x)
            loss = F.dice_loss(input=predictions, label=label)
    """
    assert input.dtype in (paddle.float32, paddle.float64)
    assert label.dtype in (paddle.int32, paddle.int64)
    assert (
        len(input.shape) >= 2
    ), "The rank of input should be greater than or equal to 2."
    assert len(input.shape) == len(label.shape), (
        "The rank of input and label should be equal, "
        "but received input: %d, label: %d."
        % (len(input.shape), len(label.shape))
    )
    assert label.shape[-1] == 1, (
        "The last dimension of label should be 1, "
        "but received %d." % label.shape[-1]
    )
    assert (
        input.shape[:-1] == label.shape[:-1]
    ), "All dimensions should be equal except the last one."
    assert (
        input.numel() > 0 and label.numel() > 0
    ), "Any dimension of input and label cannot be equal to 0."

    label = paddle.squeeze(label, [-1])
    label = paddle.nn.functional.one_hot(label, input.shape[-1])
    reduce_dim = list(range(1, len(input.shape)))
    inse = paddle.sum(input * label, axis=reduce_dim)
    dice_denominator = paddle.sum(input, axis=reduce_dim) + paddle.sum(
        label, axis=reduce_dim
    )
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    return paddle.mean(dice_score)


def log_loss(input, label, epsilon=1e-4, name=None):
    r"""

    **Negative Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    negative log loss.

    .. math::

        Out = -label * \log{(input + \epsilon)}
              - (1 - label) * \log{(1 - input + \epsilon)}

    Args:
        input (Tensor|list):  A 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator. Data type float32.
        label (Tensor|list):  The ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size.
                                Data type float32.
        epsilon (float, optional): A small number for numerical stability. Default 1e-4.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Tensor, which shape is [N x 1], data type is float32.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn.functional as F

          label = paddle.randn((10,1))
          prob = paddle.randn((10,1))
          cost = F.log_loss(input=prob, label=label)
    """
    if in_dygraph_mode():
        return _C_ops.log_loss(input, label, epsilon)

    helper = LayerHelper('log_loss', **locals())
    check_variable_and_dtype(input, 'input', ['float32'], 'log_loss')
    check_variable_and_dtype(label, 'label', ['float32'], 'log_loss')

    loss = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='log_loss',
        inputs={'Predicted': [input], 'Labels': [label]},
        outputs={'Loss': [loss]},
        attrs={'epsilon': epsilon},
    )
    return loss


def fluid_softmax_with_cross_entropy(
    logits,
    label,
    soft_label=False,
    ignore_index=-100,
    numeric_stable_mode=True,
    return_softmax=False,
    axis=-1,
):
    r"""

    This operator implements the cross entropy loss function with softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators
    expects mutually exclusive hard labels, each sample in a batch is in exactly
    one class with a probability of 1.0. Each sample in the batch will have a
    single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::
        \\loss_j=-\text{logits}_{label_j} +\log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::
        \\loss_j= -\sum_{i=0}^{K}\text{label}_i\left(\text{logits}_i - \log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right)\right), j = 1,...,K

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::
        \\max_j&=\max_{i=0}^{K}{\text{logits}_i} \\
                log\_max\_sum_j &= \log\sum_{i=0}^{K}\exp(logits_i - max_j)\\
                softmax_j &= \exp(logits_j - max_j - {log\_max\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64. The input tensor of unscaled log probabilities.
        label (Tensor): The ground truth  ``Tensor`` , data type is the same
            as the ``logits`` . If :attr:`soft_label` is set to :attr:`True`,
            Label is a ``Tensor``  in the same shape with :attr:`logits`.
            If :attr:`soft_label` is set to :attr:`True`, Label is a ``Tensor``
            in the same shape with :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool, optional): A flag to indicate whether to interpretant the given
            labels as soft labels. Default False.
        ignore_index (int, optional): Specifies a target value that is ignored and does
                                      not contribute to the input gradient. Only valid
                                      if :attr:`soft_label` is set to :attr:`False`.
                                      Default: kIgnoreIndex(-100).
        numeric_stable_mode (bool, optional): A flag to indicate whether to use a more
                                              numerically stable algorithm. Only valid
                                              when :attr:`soft_label` is :attr:`False`
                                              and GPU is used. When :attr:`soft_label`
                                              is :attr:`True` or CPU is used, the
                                              algorithm is always numerically stable.
                                              Note that the speed may be slower when use
                                              stable algorithm. Default: True.
        return_softmax (bool, optional): A flag indicating whether to return the softmax
                                         along with the cross entropy loss. Default: False.
        axis (int, optional): The index of dimension to perform softmax calculations. It
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.

    Returns:
        ``Tensor`` or Tuple of two ``Tensor`` : Return the cross entropy loss if \
                                                    `return_softmax` is False, otherwise the tuple \
                                                    (loss, softmax), softmax is in the same shape \
                                                    with input logits and cross entropy loss is in \
                                                    the same shape with input logits except shape \
                                                    in dimension :attr:`axis` as 1.

    Examples:
        .. code-block:: python

            import paddle

            logits = paddle.to_tensor([0.4, 0.6, 0.9])
            label = paddle.randint(high=2, shape=[1], dtype="int64")

            out = paddle.nn.functional.softmax_with_cross_entropy(logits=logits, label=label)
            print(out)
            # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1.15328646])
    """
    if _non_static_mode():
        if core.is_compiled_with_npu():
            softmax, backprop, loss = _legacy_C_ops.softmax_with_cross_entropy(
                logits,
                label,
                'soft_label',
                soft_label,
                'ignore_index',
                ignore_index,
                'numeric_stable_mode',
                numeric_stable_mode,
                'axis',
                axis,
            )
        else:
            if in_dygraph_mode():
                softmax, loss = _C_ops.cross_entropy_with_softmax(
                    logits,
                    label,
                    soft_label,
                    True,
                    numeric_stable_mode,
                    ignore_index,
                    axis,
                )
            if _in_legacy_dygraph():
                softmax, loss = _legacy_C_ops.softmax_with_cross_entropy(
                    logits,
                    label,
                    'soft_label',
                    soft_label,
                    'ignore_index',
                    ignore_index,
                    'numeric_stable_mode',
                    numeric_stable_mode,
                    'axis',
                    axis,
                )
        if not return_softmax:
            return loss
        else:
            return loss, softmax

    attrs = {
        'soft_label': soft_label,
        'ignore_index': ignore_index,
        'numeric_stable_mode': numeric_stable_mode,
        'axis': axis,
    }
    helper = LayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)

    outputs = {'Softmax': softmax, 'Loss': loss}
    if core.is_compiled_with_npu() or core.is_compiled_with_mlu():
        backprop = helper.create_variable_for_type_inference(dtype=logits.dtype)
        outputs['Backprop'] = backprop
    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': logits, 'Label': label},
        outputs=outputs,
        attrs=attrs,
    )

    if return_softmax:
        return loss, softmax

    return loss


def npair_loss(anchor, positive, labels, l2_reg=0.002):
    """

    Npair loss requires paired data. Npair loss has two parts: the first part is L2
    regularizer on the embedding vector; the second part is cross entropy loss which
    takes the similarity matrix of anchor and positive as logits.

    For more information, please refer to:
    `Improved Deep Metric Learning with Multi class N pair Loss Objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_

    Args:
      anchor(Tensor): embedding vector for the anchor image. shape=[batch_size, embedding_dims],
                        the data type is float32 or float64.
      positive(Tensor): embedding vector for the positive image. shape=[batch_size, embedding_dims],
                        the data type is float32 or float64.
      labels(Tensor): 1-D tensor. shape=[batch_size], the data type is float32 or float64 or int64.
      l2_reg(float32): L2 regularization term on embedding vector, default: 0.002.


    Returns:
      A Tensor representing the npair loss, the data type is the same as anchor, the shape is [1].

    Examples:

      .. code-block:: python

          import paddle

          DATATYPE = "float32"

          anchor = paddle.rand(shape=(18, 6), dtype=DATATYPE)
          positive = paddle.rand(shape=(18, 6), dtype=DATATYPE)
          labels = paddle.rand(shape=(18,), dtype=DATATYPE)

          npair_loss = paddle.nn.functional.npair_loss(anchor, positive, labels, l2_reg = 0.002)
          print(npair_loss)

    """
    check_variable_and_dtype(
        anchor, 'anchor', ['float32', 'float64'], 'npair_loss'
    )
    check_variable_and_dtype(
        positive, 'positive', ['float32', 'float64'], 'positive'
    )
    check_variable_and_dtype(
        labels, 'labels', ['float32', 'float64', 'int64'], 'labels'
    )
    Beta = 0.25
    batch_size = labels.shape[0]

    labels = paddle.reshape(labels, shape=[batch_size, 1])
    labels = paddle.tile(labels, repeat_times=[1, batch_size])

    labels = paddle.equal(labels, paddle.transpose(labels, perm=[1, 0])).astype(
        'float32'
    )
    labels = labels / paddle.sum(labels, axis=1, keepdim=True)

    l2loss = paddle.mean(paddle.sum(paddle.square(anchor), 1)) + paddle.mean(
        paddle.sum(paddle.square(positive), 1)
    )
    l2loss = l2loss * Beta * l2_reg

    similarity_matrix = paddle.matmul(
        anchor, positive, transpose_x=False, transpose_y=True
    )
    softmax_ce = fluid_softmax_with_cross_entropy(
        logits=similarity_matrix, label=labels, soft_label=True
    )
    cross_entropy = paddle.sum(labels * softmax_ce, 0)
    celoss = paddle.mean(cross_entropy)

    return l2loss + celoss


def square_error_cost(input, label):
    r"""

    This op accepts input predictions and target label and returns the
    squared error cost.

    For predictions label, and target label, the equation is:

    .. math::

        Out = (input - label)^2

    Parameters:
        input (Tensor): Input tensor, the data type should be float32.
        label (Tensor): Label tensor, the data type should be float32.

    Returns:
        Tensor, The tensor storing the element-wise squared error
        difference between input and label.

    Examples:

        .. code-block:: python

            import paddle
            input = paddle.to_tensor([1.1, 1.9])
            label = paddle.to_tensor([1.0, 2.0])
            output = paddle.nn.functional.square_error_cost(input, label)
            print(output)
            # [0.01, 0.01]

    """
    if in_dygraph_mode():
        minus_out = _C_ops.subtract(input, label)
        square_out = _C_ops.square(minus_out)
        return square_out
    elif _in_legacy_dygraph():
        minus_out = _legacy_C_ops.elementwise_sub(input, label)
        square_out = _legacy_C_ops.square(minus_out)
        return square_out

    check_variable_and_dtype(
        input, "input", ['float32', 'float64'], 'square_error_cost'
    )
    check_variable_and_dtype(
        label, "label", ['float32', 'float64'], 'square_error_cost'
    )
    helper = LayerHelper('square_error_cost', **locals())
    minus_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input], 'Y': [label]},
        outputs={'Out': [minus_out]},
    )

    square_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='square', inputs={'X': [minus_out]}, outputs={'Out': [square_out]}
    )
    return square_out


def edit_distance(
    input,
    label,
    normalized=True,
    ignored_tokens=None,
    input_length=None,
    label_length=None,
):
    """
    This op computes the edit distances, also called Levenshtein distance, between a batch of
    hypothesis strings and their references. It measures how dissimilar two strings are by counting
    the minimum number of operations to transform one string into another.
    The operations include insertion, deletion, and substitution.

    For example, given hypothesis string A = "kitten" and reference
    B = "sitting", A will be transformed into B
    at least after two substitutions and one insertion:

    "kitten" -> "sitten" -> "sittin" -> "sitting"

    So the edit distance between A and B is 3.

    The input is a Tensor, the input_length and label_length should be supported.

    The `batch_size` of labels should be same as `input`.

    The output include the edit distance value between every pair of input and related label, and the number of sequence.
    If Attr(normalized) is true,
    the edit distance value will be divided by the length of label.

    Parameters:
        input(Tensor): The input tensor, its rank should be equal to 2 and its data type should be int64.
        label(Tensor): The label tensor, its rank should be equal to 2 and its data type should be int64.
        normalized(bool, default True): Indicated whether to normalize the edit distance.
        ignored_tokens(list<int>, default None): Tokens that will be removed before
                                     calculating edit distance.
        input_length(Tensor): The length for each sequence in `input` if it's of Tensor type, it should have shape `(batch_size, )` and its data type should be int64.
        label_length(Tensor): The length for each sequence in `label` if it's of Tensor type, it should have shape `(batch_size, )` and its data type should be int64.
        NOTE: To be avoid unexpected result, the value of every elements in input_length and label_length should be equal to the value of the second dimension of input and label. For example, The input: [[1,2,3,4],[5,6,7,8],[9,10,11,12]], the shape of input is [3,4] and the input_length should be [4,4,4]

    Returns:
        Tuple:
            distance(Tensor): edit distance result, its data type is float32, and its shape is (batch_size, 1).
            sequence_num(Tensor): sequence number, its data type is float32, and its shape is (1,).

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.to_tensor([[1,2,3],[4,5,6],[4,4,4],[1,1,1]], dtype='int64')
            label = paddle.to_tensor([[1,3,4,1],[4,5,8,1],[7,7,7,1],[1,1,1,1]], dtype='int64')
            input_len = paddle.to_tensor([3,3,3,3], dtype='int64')
            label_len = paddle.to_tensor([4,4,4,4], dtype='int64')

            distance, sequence_num = F.loss.edit_distance(input=input, label=label, input_length=input_len, label_length=label_len, normalized=False)

            # print(distance)
            # [[3.]
            #  [2.]
            #  [4.]
            #  [1.]]
            # if set normalized to True
            # [[0.75]
            #  [0.5 ]
            #  [1.  ]
            #  [0.25]
            #
            # print(sequence_num)
            # [4]

    """
    check_variable_and_dtype(input, 'input', ['int64'], 'edit_distance')
    check_variable_and_dtype(label, 'label', ['int64'], 'edit_distance')
    helper = LayerHelper("edit_distance", **locals())

    # remove some tokens from input and labels
    if ignored_tokens is not None and len(ignored_tokens) > 0:
        erased_input = helper.create_variable_for_type_inference(dtype="int64")
        erased_label = helper.create_variable_for_type_inference(dtype="int64")

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [input]},
            outputs={"Out": [erased_input]},
            attrs={"tokens": ignored_tokens},
        )
        input = erased_input

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [label]},
            outputs={"Out": [erased_label]},
            attrs={"tokens": ignored_tokens},
        )
        label = erased_label

    if in_dygraph_mode():
        return _C_ops.edit_distance(
            input, label, input_length, label_length, normalized
        )

    this_inputs = {"Hyps": [input], "Refs": [label]}
    if input_length is not None and label_length is not None:
        this_inputs['HypsLength'] = [input_length]
        this_inputs['RefsLength'] = [label_length]

    # edit distance op
    edit_distance_out = helper.create_variable_for_type_inference(dtype="int64")
    sequence_num = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type="edit_distance",
        inputs=this_inputs,
        outputs={"Out": [edit_distance_out], "SequenceNum": [sequence_num]},
        attrs={"normalized": normalized},
    )

    return edit_distance_out, sequence_num


def binary_cross_entropy(
    input, label, weight=None, reduction='mean', name=None
):
    """
    Measure the binary_cross_entropy loss between input predictions ``input``
    and target labels ``label`` . The binary_cross_entropy loss can be described as:

    If :attr:`weight` is set, the loss is:

    .. math::
        Out = -1 * weight * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`weight` is None, the loss is:

    .. math::
        Out = -1 * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`reduction` set to ``'none'``, the interface will return the original loss `Out`.

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(Out)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(Out)

    Note that the input predictions ``input`` always be the output of sigmoid, and the target labels ``label``
    should be numbers between 0 and 1.

    Parameters:
        input (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``input``
            should always be the output of sigmod.  Available dtype is float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``input``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float32, float64.
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, has to be a Tensor of size nbatch and the data type
            is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor. If ``reduction`` is ``'none'``, the shape of output is
            same as ``input`` , else the shape of output is scalar.

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.to_tensor([0.5, 0.6, 0.7], 'float32')
            label = paddle.to_tensor([1.0, 0.0, 1.0], 'float32')
            output = paddle.nn.functional.binary_cross_entropy(input, label)
            print(output)  # [0.65537095]

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in binary_cross_entropy should be 'sum', "
            "'mean' or 'none', but received %s, which is not allowed."
            % reduction
        )

    if in_dygraph_mode():
        out = _C_ops.bce_loss(input, label)
        if weight is not None:
            out = _C_ops.multiply(out, weight, 'axis', -1)

        if reduction == 'sum':
            return _C_ops.sum(out, [], None, False)

        elif reduction == 'mean':
            return _C_ops.mean_all(out)
        else:
            return out
    else:
        if _in_legacy_dygraph():
            out = _legacy_C_ops.bce_loss(input, label)
            if weight is not None:
                out = _legacy_C_ops.elementwise_mul(out, weight, 'axis', -1)
            if reduction == 'sum':
                return _legacy_C_ops.reduce_sum(
                    out, 'dim', [0], 'keep_dim', False, "reduce_all", True
                )
            elif reduction == 'mean':
                return _legacy_C_ops.mean(out)
            else:
                return out
        else:
            check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], 'binary_cross_entropy'
            )
            check_variable_and_dtype(
                label, 'label', ['float32', 'float64'], 'binary_cross_entropy'
            )

            sub_name = name if weight is None and reduction == 'none' else None
            helper = LayerHelper("binary_cross_entropy", name=sub_name)
            out = helper.create_variable_for_type_inference(dtype=input.dtype)
            helper.append_op(
                type='bce_loss',
                inputs={
                    'X': [input],
                    'Label': [label],
                },
                outputs={'Out': [out]},
            )

            if weight is not None:
                if isinstance(weight, paddle.static.Variable):
                    weight_name = name if reduction == 'none' else None
                    out = paddle.multiply(out, weight, name=weight_name)
                else:
                    raise ValueError(
                        "The weight is not a Tensor, please convert to Tensor."
                    )

            if reduction == 'sum':
                return paddle.sum(out, name=name)
            elif reduction == 'mean':
                return paddle.mean(out, name=name)
            else:
                return out


def binary_cross_entropy_with_logits(
    logit, label, weight=None, reduction='mean', pos_weight=None, name=None
):
    r"""
    Combine the sigmoid layer and the :ref:`api_nn_loss_BCELoss` layer.

    This measures the element-wise probability error in classification tasks
    in which each class is independent.
    This can be thought of as predicting labels for a data-point, where labels
    are not mutually exclusive. For example, a news article can be about
    politics, technology or sports at the same time or none of these.

    Firstly, calculate loss function as follows:

    .. math::
           Out = -Labels * \log(\sigma(Logit)) - (1 - Labels) * \log(1 - \sigma(Logit))

    We know that :math:`\sigma(Logit) = \frac{1}{1 + e^{-Logit}}`. By substituting this we get:

    .. math::
           Out = Logit - Logit * Labels + \log(1 + e^{-Logit})

    For stability and to prevent overflow of :math:`e^{-Logit}` when Logit < 0,
    we reformulate the loss as follows:

    .. math::
           Out = \max(Logit, 0) - Logit * Labels + \log(1 + e^{-\|Logit\|})

    Then, if ``weight`` or ``pos_weight`` is not None, then multiply the
    weight tensor on the loss `Out`. The ``weight`` tensor will attach different
    weight on every items in the batch. The ``pos_weight`` will attach different
    weight on the positive label of each class.

    Finally, apply reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target labels ``label`` should be numbers between 0 and 1.

    Args:
        logit (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``logit``
            is usually the output of Linear layer. Available dtype is float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``logit``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float32, float64.
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, it has to be a 1D Tensor whose size is `[N, ]`,
            The data type is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector
            with length equal to the number of classes. The data type is float32, float64.
            Default is ``'None'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. If ``reduction`` is ``'none'``, the shape of output is
            same as ``logit`` , else the shape of output is scalar.

    Examples:

        .. code-block:: python

            import paddle

            logit = paddle.to_tensor([5.0, 1.0, 3.0])
            label = paddle.to_tensor([1.0, 0.0, 1.0])
            output = paddle.nn.functional.binary_cross_entropy_with_logits(logit, label)
            print(output)  # [0.45618808]

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in binary_cross_entropy_with_logits "
            "should be 'sum', 'mean' or 'none', but received %s, which is not allowed."
            % reduction
        )

    if in_dygraph_mode():
        one = _C_ops.full(
            [1],
            float(1.0),
            core.VarDesc.VarType.FP32,
            _current_expected_place(),
        )
        out = _C_ops.sigmoid_cross_entropy_with_logits(
            logit, label, False, -100
        )
        if pos_weight is not None:
            log_weight = _C_ops.add(
                _C_ops.multiply(label, _C_ops.subtract(pos_weight, one)), one
            )
            out = _C_ops.multiply(out, log_weight)
        if weight is not None:
            out = _C_ops.multiply(out, weight)

        if reduction == "sum":
            return _C_ops.sum(out, [], None, False)
        elif reduction == "mean":
            return _C_ops.mean_all(out)
        else:
            return out
    elif _in_legacy_dygraph():
        one = _varbase_creator(dtype=logit.dtype)
        _legacy_C_ops.fill_constant(
            one,
            'value',
            float(1.0),
            'force_cpu',
            False,
            'dtype',
            one.dtype,
            'str_value',
            '1.0',
            'shape',
            [1],
        )
        out = _legacy_C_ops.sigmoid_cross_entropy_with_logits(logit, label)
        if pos_weight is not None:
            log_weight = _legacy_C_ops.elementwise_add(
                _legacy_C_ops.elementwise_mul(
                    label, _legacy_C_ops.elementwise_sub(pos_weight, one)
                ),
                one,
            )
            out = _legacy_C_ops.elementwise_mul(out, log_weight)
        if weight is not None:
            out = _legacy_C_ops.elementwise_mul(out, weight)

        if reduction == "sum":
            return _legacy_C_ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == "mean":
            return _legacy_C_ops.mean(out)
        else:
            return out

    check_variable_and_dtype(
        logit,
        'logit',
        ['float32', 'float64'],
        'binary_cross_entropy_with_logits',
    )
    check_variable_and_dtype(
        label,
        'label',
        ['float32', 'float64'],
        'binary_cross_entropy_with_logits',
    )
    sigmoid_name = None
    if reduction == 'none' and pos_weight is None and weight is None:
        sigmoid_name = name

    helper = LayerHelper("sigmoid_cross_entropy_with_logits", **locals())

    out = helper.create_variable_for_type_inference(dtype=logit.dtype)

    helper.append_op(
        type="sigmoid_cross_entropy_with_logits",
        inputs={"X": logit, "Label": label},
        attrs={"ignore_index": kIgnoreIndex, 'normalize': False},
        outputs={"Out": out},
    )

    one = paddle.full(shape=[1], fill_value=1.0, dtype=logit.dtype)
    if pos_weight is not None:
        check_variable_and_dtype(
            pos_weight,
            'pos_weight',
            ['float32', 'float64'],
            'binary_cross_entropy_with_logits',
        )
        log_weight = paddle.add(
            paddle.multiply(label, paddle.subtract(pos_weight, one)), one
        )
        pos_weight_name = (
            name if reduction == 'none' and weight is None else None
        )
        out = paddle.multiply(out, log_weight, name=pos_weight_name)

    if weight is not None:
        check_variable_and_dtype(
            weight,
            'weight',
            ['float32', 'float64'],
            'binary_cross_entropy_with_logits',
        )
        weight_name = name if reduction == 'none' else None
        out = paddle.multiply(out, weight, name=weight_name)

    if reduction == "sum":
        return paddle.sum(out, name=name)
    elif reduction == "mean":
        return paddle.mean(out, name=name)
    return out


def hsigmoid_loss(
    input,
    label,
    num_classes,
    weight,
    bias=None,
    path_table=None,
    path_code=None,
    is_sparse=False,
    name=None,
):
    """
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.

    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.

    Comparing to softmax, hsigmoid can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The API supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_.

    For the custom tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        input (Tensor): A tensor with the shape [N, D], where N is the size of mini-batch,
            and D is the feature size. Its data type supports float32 or float64.
        label (Tensor): A tensor contains the labels of training data. Its shape is [N, 1]
            and data type is int64.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (path_code and path_table is None are None), `num_classes`
            should not be None. If the custom tree is used (path_code and path_table is None are not None),
            `num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        weight (Tensor): A tensor with shape (num_classes - 1, D), with the same data type as `input`.
        bias (Tensor, optional): A tensor with shape (num_classes - 1, 1), with the same data type as `input`.
            If `bias` is None, no bias will be add. Default is None.
        path_table (Tensor, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. If `path_table` and `path_code` are None, the default tree will be used.
            Default is None.
        path_code (Tensor, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. If `path_table` and
            `path_code` are None, the default tree will be used. Default is None.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating. If `is_sparse` is True,
            the gradient of `weight` and `input` will be sparse. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as `input`.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            paddle.set_device('cpu')

            input = paddle.uniform([4, 3])
            # [[0.45424712  -0.77296764  0.82943869] # random
            #  [0.85062802  0.63303483  0.35312140] # random
            #  [0.57170701  0.16627562  0.21588242] # random
            #  [0.27610803  -0.99303514  -0.17114788]] # random
            label = paddle.to_tensor([0, 1, 4, 5])
            num_classes = 5
            weight=paddle.uniform([num_classes-1, 3])
            # [[-0.64477652  0.24821866  -0.17456549] # random
            #  [-0.04635394  0.07473493  -0.25081766] # random
            #  [ 0.05986035  -0.12185556  0.45153677] # random
            #  [-0.66236806  0.91271877  -0.88088769]] # random

            out=F.hsigmoid_loss(input, label, num_classes, weight)
            # [[1.96709502]
            #  [2.40019274]
            #  [2.11009121]
            #  [1.92374969]]
    """
    if in_dygraph_mode():
        out, _, _ = _C_ops.hsigmoid_loss(
            input,
            label,
            weight,
            bias,
            path_table,
            path_code,
            num_classes,
            is_sparse,
            is_sparse,
        )
        return out
    elif _in_legacy_dygraph():
        out, _, _ = _legacy_C_ops.hierarchical_sigmoid(
            input,
            weight,
            label,
            path_table,
            path_code,
            bias,
            'num_classes',
            num_classes,
            'is_sparse',
            is_sparse,
            'remote_prefetch',
            is_sparse,
        )
        return out

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'hsigmoid_loss'
    )
    check_variable_and_dtype(label, 'label', ['int64'], 'hsigmoid_loss')
    check_variable_and_dtype(
        weight, 'weight', ['float32', 'float64'], 'hsigmoid_loss'
    )
    if bias is not None:
        check_variable_and_dtype(
            bias, 'bias', ['float32', 'float64'], 'hsigmoid_loss'
        )
    if path_table is not None:
        check_variable_and_dtype(
            path_table, 'path_table', ['int64'], 'hsigmoid_loss'
        )
    if path_code is not None:
        check_variable_and_dtype(
            path_code, 'path_code', ['int64'], 'hsigmoid_loss'
        )

    attrs = {
        "num_classes": num_classes,
        "is_sparse": is_sparse,
        "remote_prefetch": is_sparse,
    }

    inputs = {
        "X": input,
        "W": weight,
        "Bias": bias,
        "PathTable": path_table,
        "PathCode": path_code,
        "Label": label,
    }

    helper = LayerHelper('hsigmoid_loss', **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    pre_out = helper.create_variable_for_type_inference(input.dtype)
    outputs = {"Out": out, "PreOut": pre_out, "W_Out": weight}

    helper.append_op(
        type="hierarchical_sigmoid", inputs=inputs, outputs=outputs, attrs=attrs
    )
    return out


def smooth_l1_loss(input, label, reduction='mean', delta=1.0, name=None):
    r"""
    Calculate smooth_l1_loss. Creates a criterion that uses a squared
    term if the absolute element-wise error falls below 1 and an L1 term otherwise.
    In some cases it can prevent exploding gradients and it is more robust and less
    sensitivity to outliers. Also known as the Huber loss:

    .. math::

        loss(x,y) = \frac{1}{n}\sum_{i}z_i


    where :math:`z_i` is given by:

    .. math::

        \mathop{z_i} = \left\{\begin{array}{rcl}
                0.5(x_i - y_i)^2 & & {if |x_i - y_i| < \delta} \\
                \delta * |x_i - y_i| - 0.5 * \delta^2 & & {otherwise}
            \end{array} \right.

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is
            (N, C), where C is number of classes, and if shape is more than 2D, this
            is (N, C, D1, D2,..., Dk), k >= 1.
        label (Tensor): Label tensor, the data type is float32 or float64. The shape of label
            is the same as the shape of input.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        delta (float, optional): Specifies the hyperparameter :math:`\delta` to be used.
            The value determines how large the errors need to be to use L1. Errors
            smaller than delta are minimized with L2. Parameter is ignored for
            negative/zero values. Default = 1.0
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, The tensor variable storing the smooth_l1_loss of input and label.

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.rand([3, 3]).astype('float32')
            label = paddle.rand([3, 3]).astype('float32')
            output = paddle.nn.functional.smooth_l1_loss(input, label)
            print(output)
            # [0.068004]
    """
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'smooth_l1_loss'
    )
    check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'smooth_l1_loss'
    )

    if in_dygraph_mode():
        out, residual = _C_ops.huber_loss(input, label, delta)
    else:
        helper = LayerHelper('huber_loss', **locals())
        residual = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
        helper.append_op(
            type='huber_loss',
            inputs={'X': input, 'Y': label},
            outputs={'Out': out, 'Residual': residual},
            attrs={'delta': delta},
        )

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in smooth_l1_loss should be 'sum', 'mean' or"
            " 'none', but received %s, which is not allowed." % reduction
        )
    if reduction == 'none':
        return out
    elif reduction == 'mean':
        return paddle.mean(out)
    elif reduction == 'sum':
        return paddle.sum(out)


def margin_ranking_loss(
    input, other, label, margin=0.0, reduction='mean', name=None
):
    r"""

    Calcluate the margin rank loss between the input, other and label, use the math function as follows.

    .. math::
        margin\_rank\_loss = max(0, -label * (input - other) + margin)

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(margin\_rank\_loss)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(margin\_rank\_loss)

    If :attr:`reduction` set to ``'none'``, just return the origin ``margin_rank_loss``.

    Parameters:
        input(Tensor): the first input tensor, it's data type should be float32, float64.
        other(Tensor): the second input tensor, it's data type should be float32, float64.
        label(Tensor): the label value corresponding to input, it's data type should be float32, float64.
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candicates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[1]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Examples:

        .. code-block:: python

            import paddle

            input = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
            other = paddle.to_tensor([[2, 1], [2, 4]], dtype='float32')
            label = paddle.to_tensor([[1, -1], [-1, -1]], dtype='float32')
            loss = paddle.nn.functional.margin_ranking_loss(input, other, label)
            print(loss) # [0.75]
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in MarginRankingLoss should be 'sum', 'mean' or 'none', but "
            "received %s, which is not allowed." % reduction
        )
    if in_dygraph_mode():
        out = _C_ops.subtract(other, input)
        out = _C_ops.multiply(out, label)
        if margin != 0.0:
            margin = fluid.dygraph.base.to_variable([margin], dtype=out.dtype)
            out = _C_ops.add(out, margin)
        out = _C_ops.relu(out)
        if reduction == 'sum':
            return _C_ops.sum(out, [], None, False)
        elif reduction == 'mean':
            return _C_ops.mean_all(out)
        return out
    elif _in_legacy_dygraph():
        out = _legacy_C_ops.elementwise_sub(other, input)
        out = _legacy_C_ops.elementwise_mul(out, label)
        if margin != 0.0:
            margin = fluid.dygraph.base.to_variable([margin], dtype=out.dtype)
            out = _legacy_C_ops.elementwise_add(out, margin)
        out = _legacy_C_ops.relu(out)
        if reduction == 'sum':
            return _legacy_C_ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == 'mean':
            return _legacy_C_ops.mean(out)
        return out

    helper = LayerHelper("margin_ranking_loss", **locals())
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'margin_rank_loss'
    )
    check_variable_and_dtype(
        other, 'other', ['float32', 'float64'], 'margin_rank_loss'
    )
    check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'margin_rank_loss'
    )

    out = paddle.subtract(input, other)
    neg_label = paddle.neg(label)
    out = paddle.multiply(neg_label, out)

    if margin != 0.0:
        margin_var = out.block.create_var(dtype=out.dtype)
        margin_var = paddle.full(shape=[1], fill_value=margin, dtype=out.dtype)
        out = paddle.add(out, margin_var)

    result_out = helper.create_variable_for_type_inference(input.dtype)

    if reduction == 'none':
        helper.append_op(
            type="relu", inputs={"X": out}, outputs={"Out": result_out}
        )
        return result_out
    elif reduction == 'sum':
        out = paddle.nn.functional.relu(out)
        attrs = {"dim": [0], "keep_dim": False, "reduce_all": True}
        helper.append_op(
            type="reduce_sum",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs=attrs,
        )
        return result_out
    elif reduction == 'mean':
        out = paddle.nn.functional.relu(out)
        helper.append_op(
            type="mean",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs={},
        )
        return result_out


def l1_loss(input, label, reduction='mean', name=None):
    r"""

    Computes the L1 Loss of Tensor ``input`` and ``label`` as follows.

    If `reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \lvert input - label \rvert

    If `reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\lvert input - label \rvert)

    If `reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\lvert input - label \rvert)


    Parameters:
        input (Tensor): The input tensor. The shapes is [N, `*`], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, `*`], same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the L1 Loss of Tensor ``input`` and ``label``.
        If `reduction` is ``'none'``, the shape of output loss is :math:`[N, *]`, the same as ``input`` .
        If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1].

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.to_tensor([[1.5, 0.8], [0.2, 1.3]])
            label = paddle.to_tensor([[1.7, 1], [0.4, 0.5]])

            l1_loss = paddle.nn.functional.l1_loss(input, label)
            print(l1_loss)
            # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [0.34999999])

            l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='none')
            print(l1_loss)
            # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [[0.20000005, 0.19999999],
            #         [0.20000000, 0.79999995]])

            l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='sum')
            print(l1_loss)
            # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1.39999998])

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
            "received %s, which is not allowed." % reduction
        )

    if in_dygraph_mode():
        unreduced = _C_ops.abs(_C_ops.subtract(input, label))

        if reduction == 'mean':
            return _C_ops.mean_all(unreduced)
        elif reduction == 'sum':
            return _C_ops.sum(unreduced, [], None, False)
        else:
            return unreduced
    elif _in_legacy_dygraph():
        unreduced = _elementwise_op_in_dygraph(
            input, label, axis=-1, act='abs', op_name='elementwise_sub'
        )
        if reduction == 'mean':
            return _legacy_C_ops.mean(unreduced)
        elif reduction == 'sum':
            return _legacy_C_ops.reduce_sum(
                unreduced, 'dim', [0], 'keep_dim', False, 'reduce_all', True
            )
        else:
            return unreduced

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'l1_loss'
    )
    check_variable_and_dtype(
        label, 'label', ['float32', 'float64', 'int32', 'int64'], 'l1_loss'
    )

    if reduction == 'sum':
        unreduced = paddle.abs(paddle.subtract(x=input, y=label))
        return paddle.sum(unreduced, name=name)
    elif reduction == 'mean':
        unreduced = paddle.abs(paddle.subtract(x=input, y=label))
        return paddle.mean(unreduced, name=name)
    else:
        return paddle.abs(paddle.subtract(x=input, y=label, name=name))


def nll_loss(
    input, label, weight=None, ignore_index=-100, reduction='mean', name=None
):
    """
    This api returns negative log likelihood.
    See more detail in :ref:`NLLLoss <api_paddle_nn_NLLLoss>` .


    Parameters:
         input (Tensor): Input tensor, the shape is :math:`[N, C]`, `C` is the number of classes.
             But in K-dimension situation, the shape is :math:`[N, C, d_1, d_2, ..., d_K]`.
             The data type is float32, float64.
         label (Tensor): Label tensor, the shape is :math:`[N,]` or :math:`[N, d_1, d_2, ..., d_K]`.
             The data type is int64.
         weight (Tensor, optional): Weight tensor, a manual rescaling weight given
             to each class. If given, it has to be a 1D Tensor whose size is `[C, ]`. Otherwise,
             it treated as if having all ones. the data type is
             float32, float64, Default is ``'None'``.
         ignore_index (int, optional): Specifies a target value that is ignored
             and does not contribute to the input gradient. Default is -100.
         reduction (str, optional): Indicate how to average the loss,
             the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
             If `reduction` is ``'mean'``, the reduced mean loss is returned;
             if `reduction` is ``'sum'``, the reduced sum loss is returned;
             if `reduction` is ``'none'``, no reduction will be apllied.
             Default is ``'mean'``.
         name (str, optional): Name for the operation (optional, default is None).
             For more information, please refer to :ref:`api_guide_Name`.

    Returns:
         `Tensor`, the value of negative log likelihood loss.

    Examples:
        .. code-block:: python

                import paddle
                from paddle.nn.functional import nll_loss
                log_softmax = paddle.nn.LogSoftmax(axis=1)

                input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
                          [0.53331435, 0.07999352, 0.8549948 ],
                          [0.25879037, 0.39530203, 0.698465  ],
                          [0.73427284, 0.63575995, 0.18827209],
                          [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
                log_out = log_softmax(input)
                label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
                result = nll_loss(log_out, label)
                print(result) # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True, [1.07202101])
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
            "'none', but received %s, which is not allowed." % reduction
        )

    input_shape = list(input.shape)
    input_dims = len(input_shape)
    if input_dims < 2:
        raise ValueError(
            'Expected 2 or more dimensions (got {})'.format(input_dims)
        )
    n = input_shape[0]
    c = input_shape[1]
    if in_dygraph_mode():
        if input_dims != 2 and input_dims != 4:
            input = _C_ops.reshape(input, [n, c, 1, -1])
            label = _C_ops.reshape(label, [n, 1, -1])
            out_shape = [n] + input_shape[2:]
        out, total_weight = _C_ops.nll_loss(
            input, label, weight, ignore_index, reduction
        )
        if input_dims != 2 and input_dims != 4 and reduction == 'none':
            out = _C_ops.reshape(out, out_shape)
        return out
    elif _in_legacy_dygraph():
        if input_dims != 2 and input_dims != 4:
            input, _ = _legacy_C_ops.reshape2(
                input, None, 'shape', [n, c, 1, -1]
            )
            label, _ = _legacy_C_ops.reshape2(label, None, 'shape', [n, 1, -1])
            out_shape = [n] + input_shape[2:]

        out, total_weight = _legacy_C_ops.nll_loss(
            input,
            label,
            weight,
            'ignore_index',
            ignore_index,
            'reduction',
            reduction,
        )
        if input_dims != 2 and input_dims != 4 and reduction == 'none':
            out, _ = _legacy_C_ops.reshape2(out, None, 'shape', out_shape)
        return out

    helper = LayerHelper('nll_loss', **locals())

    if input_dims != 2 and input_dims != 4:
        input = reshape(input, shape=[n, c, 1, -1])
        label = reshape(label, shape=[n, 1, -1])
        out_shape = [n] + input_shape[2:]

    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'nll_loss')
    check_variable_and_dtype(label, 'label', ['int64'], 'nll_loss')
    inputs = {'X': input, 'Label': label}
    attrs = {'reduction': reduction, 'ignore_index': ignore_index}
    if weight is not None:
        if isinstance(weight, Variable):
            inputs['Weight'] = weight

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    total_weight = helper.create_variable_for_type_inference(dtype=input.dtype)
    outputs = {'Out': out, 'Total_weight': total_weight}

    helper.append_op(
        type='nll_loss', inputs=inputs, outputs=outputs, attrs=attrs
    )
    if input_dims != 2 and input_dims != 4 and reduction == 'none':
        out = reshape(out, shape=out_shape)

    return out


def kl_div(input, label, reduction='mean', name=None):
    r"""
    Calculate the Kullback-Leibler divergence loss
    between Input(X) and Input(Target). Notes that Input(X) is the
    log-probability and Input(Target) is the probability.

    KL divergence loss is calculated as follows:

    $$l(x, y) = y * (\log(y) - x)$$

    Here :math:`x` is input and :math:`y` is label.

    If `reduction` is ``'none'``, the output loss is the same shape as the input, and the loss at each point is calculated separately. There is no reduction to the result.

    If `reduction` is ``'mean'``, the output loss is the shape of [1], and the output is the average of all losses.

    If `reduction` is ``'sum'``, the output loss is the shape of [1], and the output is the sum of all losses.

    If `reduction` is ``'batchmean'``, the output loss is the shape of [N], N is the batch size, and the output is the sum of all losses divided by the batch size.

    Args:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means
            any number of additional dimensions. It's data type should be float32, float64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64.
        reduction (str, optional): Indicate how to average the loss,
            the candicates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be apllied.
            Default is ``'mean'``.
        name(str, optional): Name for the operation (optional, default is None). For more information,
            please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The KL divergence loss. The data type is same as input tensor

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            shape = (5, 20)
            x = paddle.uniform(shape, min=-10, max=10).astype('float32')
            target = paddle.uniform(shape, min=-10, max=10).astype('float32')

            # 'batchmean' reduction, loss shape will be [1]
            pred_loss = F.kl_div(x, target, reduction='batchmean')
            # shape=[1]

            # 'mean' reduction, loss shape will be [1]
            pred_loss = F.kl_div(x, target, reduction='mean')
            # shape=[1]

            # 'sum' reduction, loss shape will be [1]
            pred_loss = F.kl_div(x, target, reduction='sum')
            # shape=[1]

            # 'none' reduction, loss shape is same with input shape
            pred_loss = F.kl_div(x, target, reduction='none')
            # shape=[5, 20]

    """
    # ugly type promotion
    if (
        fluid.data_feeder.convert_dtype(input.dtype) == 'float32'
        and fluid.data_feeder.convert_dtype(label.dtype) == 'float64'
    ):
        input = paddle.cast(input, 'float64')
    elif (
        fluid.data_feeder.convert_dtype(input.dtype) == 'float64'
        and fluid.data_feeder.convert_dtype(label.dtype) == 'float32'
    ):
        label = paddle.cast(label, 'float64')

    if in_dygraph_mode():
        out = _C_ops.kldiv_loss(input, label, 'none')
        if reduction == 'mean':
            out = paddle.mean(out)
        elif reduction == 'sum':
            out = paddle.sum(out)
        elif reduction == 'batchmean':
            if len(input.shape) > 0:
                batch_size = input.shape[0]
                out = paddle.sum(out) / batch_size
        return out
    elif _in_legacy_dygraph():
        out = _legacy_C_ops.kldiv_loss(input, label, 'reduction', 'none')
        if reduction == 'mean':
            out = paddle.mean(out)
        elif reduction == 'sum':
            out = paddle.sum(out)
        elif reduction == 'batchmean':
            if len(input.shape) > 0:
                batch_size = input.shape[0]
                out = paddle.sum(out) / batch_size
        return out

    helper = LayerHelper('kl_div', **locals())

    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'kl_div')
    check_variable_and_dtype(label, 'label', ['float32', 'float64'], 'kl_div')
    fluid.data_feeder.check_type(reduction, 'reduction', str, 'kl_div')

    loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='kldiv_loss',
        inputs={'X': input, 'Target': label},
        outputs={'Loss': loss},
        attrs={'reduction': 'none'},
    )

    if reduction == 'mean':
        loss = paddle.mean(loss)
    elif reduction == 'sum':
        loss = paddle.sum(loss)
    elif reduction == 'batchmean':
        batch_size = paddle.shape(input)[0]
        loss = paddle.sum(loss) / batch_size
    return loss


def mse_loss(input, label, reduction='mean', name=None):
    r"""
    Accept input predications and label and returns the mean square error.

    If :attr:`reduction` is set to ``'none'``, loss is calculated as:

    .. math::
        Out = (input - label)^2

    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:

    .. math::
        Out = \operatorname{mean}((input - label)^2)

    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:

    .. math::
        Out = \operatorname{sum}((input - label)^2)

    Parameters:
        input (Tensor): Input tensor, the data type should be float32 or float64.
        label (Tensor): Label tensor, the data type should be float32 or float64.
        reduction (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned.
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor, The tensor tensor storing the mean square error difference of input and label.

    Examples:

        .. code-block:: python

            import paddle
            mse_loss = paddle.nn.loss.MSELoss()
            input = paddle.to_tensor(1.5)
            label = paddle.to_tensor(1.7)
            output = mse_loss(input, label)
            print(output)
            # [0.04000002]

    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'mse_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction)
        )

    if not in_dynamic_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'mse_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'mse_loss'
        )

    if reduction == 'none':
        return paddle.square(paddle.subtract(input, label), name=name)
    elif reduction == 'mean':
        return paddle.mean(
            paddle.square(paddle.subtract(input, label)), name=name
        )
    else:
        return paddle.sum(
            paddle.square(paddle.subtract(input, label)), name=name
        )


def ctc_loss(
    log_probs,
    labels,
    input_lengths,
    label_lengths,
    blank=0,
    reduction='mean',
    norm_by_times=False,
):
    """

    An operator integrating the open source Warp-CTC library (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation
    is interated to the Warp-CTC library to normalize values for each row of the input tensor.

    Parameters:
        log_probs (Tensor): The unscaled probability sequence with padding, which is a 3-D Tensor. The tensor shape is [max_logit_length, batch_size, num_classes + 1], where max_logit_length is the longest length of input logit sequence. The data type should be float32 or float64.
        labels (Tensor): The ground truth sequence with padding, which must be a 3-D Tensor. The tensor shape is [batch_size, max_label_length], where max_label_length is the longest length of label sequence. The data type must be int32.
        input_lengths (Tensor): The length for each input sequence, it should have shape [batch_size] and dtype int64.
        label_lengths (Tensor): The length for each label sequence, it should have shape [batch_size] and dtype int64.
        blank (int, optional): The blank label index of Connectionist Temporal Classification (CTC) loss, which is in the half-opened interval [0, num_classes + 1). The data type must be int32. Default is 0.
        reduction (string, optional): Indicate how to average the loss, the candicates are ``'none'`` | ``'mean'`` | ``'sum'``. If :attr:`reduction` is ``'mean'``, the output loss will be divided by the label_lengths, and then return the mean of quotient; If :attr:`reduction` is ``'sum'``, return the sum of loss; If :attr:`reduction` is ``'none'``, no reduction will be applied. Default is ``'mean'``.
        norm_by_times (bool, default False) – Whether to normalize the gradients by the number of time-step, which is also the sequence’s length. There is no need to normalize the gradients if reduction mode is 'mean'.

    Returns:
        Tensor, The Connectionist Temporal Classification (CTC) loss between ``log_probs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is [1]. Data type is the same as ``log_probs``.

    Examples:

        .. code-block:: python

            # declarative mode
            import paddle.nn.functional as F
            import paddle

            # length of the longest logit sequence
            max_seq_length = 4
            #length of the longest label sequence
            max_label_length = 3
            # number of logit sequences
            batch_size = 2
            # class num
            class_num = 3

            log_probs = paddle.to_tensor([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
                                    [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],

                                    [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
                                    [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],

                                    [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
                                    [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],

                                    [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
                                    [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],

                                    [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
                                    [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]],
                                    dtype="float32")
            labels = paddle.to_tensor([[1, 2, 2],
                                    [1, 2, 2]], dtype="int32")
            input_lengths = paddle.to_tensor([5, 5], dtype="int64")
            label_lengths = paddle.to_tensor([3, 3], dtype="int64")

            loss = F.ctc_loss(log_probs, labels,
                input_lengths,
                label_lengths,
                blank=0,
                reduction='none')
            print(loss)
            # Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [3.91798496, 2.90765190])

            loss = F.ctc_loss(log_probs, labels,
                input_lengths,
                label_lengths,
                blank=0,
                reduction='mean')
            print(loss)
            # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1.13760614])

    """

    def warpctc(
        input,
        label,
        blank=0,
        norm_by_times=False,
        input_length=None,
        label_length=None,
    ):
        if in_dygraph_mode():
            if input_length is None or label_length is None:
                raise ValueError(
                    "input_length and label_length must not be None in dygraph mode!"
                )
            loss_out = _C_ops.warpctc(
                input, label, input_length, label_length, blank, norm_by_times
            )
            return loss_out
        if _non_static_mode():
            if input_length is None or label_length is None:
                raise ValueError(
                    "input_length and label_length must not be None in dygraph mode!"
                )
            grad, loss_out = _legacy_C_ops.warpctc(
                input,
                label,
                input_length,
                label_length,
                'blank',
                blank,
                'norm_by_times',
                norm_by_times,
            )
            return loss_out
        helper = LayerHelper('warpctc', **locals())
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], "warpctc"
        )
        check_variable_and_dtype(label, 'label', ['int32'], "warpctc")
        this_inputs = {'Logits': [input], 'Label': [label]}
        if input_length is not None and label_length is not None:
            check_variable_and_dtype(
                input_length, 'LogitsLength', ['int64'], "warpctc"
            )
            check_variable_and_dtype(
                label_length, 'LabelLength', ['int64'], "warpctc"
            )
            this_inputs['LogitsLength'] = [input_length]
            this_inputs['LabelLength'] = [label_length]

        loss_out = helper.create_variable_for_type_inference(dtype=input.dtype)
        grad_out = helper.create_variable_for_type_inference(dtype=input.dtype)

        helper.append_op(
            type='warpctc',
            inputs=this_inputs,
            outputs={'WarpCTCGrad': [grad_out], 'Loss': [loss_out]},
            attrs={
                'blank': blank,
                'norm_by_times': norm_by_times,
            },
        )
        return loss_out

    loss_out = warpctc(
        log_probs, labels, blank, norm_by_times, input_lengths, label_lengths
    )

    loss_out = paddle.squeeze(loss_out, [-1])
    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        loss_out = paddle.mean(loss_out / label_lengths)
    elif reduction == 'sum':
        loss_out = paddle.sum(loss_out)
    return loss_out


def margin_cross_entropy(
    logits,
    label,
    margin1=1.0,
    margin2=0.5,
    margin3=0.0,
    scale=64.0,
    group=None,
    return_softmax=False,
    reduction='mean',
):
    r"""
    .. math::

        L=-\frac{1}{N}\sum^N_{i=1}\log\frac{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}}{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}+\sum^n_{j=1,j\neq y_i} e^{scos\theta_{y_i}}}

    where the :math:`\theta_{y_i}` is the angle between the feature :math:`x` and
    the representation of class :math:`i`. The details of ArcFace loss
    could be referred to https://arxiv.org/abs/1801.07698.

    .. hint::
        The API supports single GPU and multi GPU, and don't supports CPU.
        For data parallel mode, set ``group=False``.
        For model parallel mode, set ``group=None`` or the group instance return by paddle.distributed.new_group.
        And logits.shape[-1] can be different at each rank.

    Args:
        logits (Tensor): shape[N, local_num_classes], the output of the normalized X multiply the normalized W.
                The logits is shard_logits when using model parallel.
        label (Tensor): shape[N] or shape[N, 1], the groud truth label.
        margin1 (float, optional): m1 of margin loss, default value is `1.0`.
        margin2 (float, optional): m2 of margin loss, default value is `0.5`.
        margin3 (float, optional): m3 of margin loss, default value is `0.0`.
        scale (float, optional): s of margin loss, default value is `64.0`.
        group (Group, optional): The group instance return by paddle.distributed.new_group
            or ``None`` for global default group or ``False`` for data parallel (do not communication cross ranks).
            Default is ``None``.
        return_softmax (bool, optional): Whether return softmax probability. Default value is `False`.
        reduction (str, optional): The candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
                    If :attr:`reduction` is ``'mean'``, return the average of loss;
                    If :attr:`reduction` is ``'sum'``, return the sum of loss;
                    If :attr:`reduction` is ``'none'``, no reduction will be applied.
                    Default value is `'mean'`.

    Returns:
        Tensor|tuple[Tensor, Tensor], return the cross entropy loss if
            `return_softmax` is False, otherwise the tuple (loss, softmax),
            softmax is shard_softmax when using model parallel, otherwise
            softmax is in the same shape with input logits. If
            ``reduction == None``, the shape of loss is ``[N, 1]``, otherwise
            the shape is ``[1]``.

    Examples:

    .. code-block:: python
        :name: code-example1

        # required: gpu
        # Single GPU
        import paddle
        m1 = 1.0
        m2 = 0.5
        m3 = 0.0
        s = 64.0
        batch_size = 2
        feature_length = 4
        num_classes = 4

        label = paddle.randint(low=0, high=num_classes, shape=[batch_size], dtype='int64')

        X = paddle.randn(
            shape=[batch_size, feature_length],
            dtype='float64')
        X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
        X = paddle.divide(X, X_l2)

        W = paddle.randn(
            shape=[feature_length, num_classes],
            dtype='float64')
        W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
        W = paddle.divide(W, W_l2)

        logits = paddle.matmul(X, W)
        loss, softmax = paddle.nn.functional.margin_cross_entropy(
            logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None)

        print(logits)
        print(label)
        print(loss)
        print(softmax)

        #Tensor(shape=[2, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #       [[ 0.85204151, -0.55557678,  0.04994566,  0.71986042],
        #        [-0.20198586, -0.35270476, -0.55182702,  0.09749021]])
        #Tensor(shape=[2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        #       [2, 3])
        #Tensor(shape=[2, 1], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #       [[82.37059586],
        #        [12.13448420]])
        #Tensor(shape=[2, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #       [[0.99978819, 0.00000000, 0.00000000, 0.00021181],
        #        [0.99992995, 0.00006468, 0.00000000, 0.00000537]])

    .. code-block:: python
        :name: code-example2

        # required: distributed
        # Multi GPU, test_margin_cross_entropy.py
        import paddle
        import paddle.distributed as dist
        strategy = dist.fleet.DistributedStrategy()
        dist.fleet.init(is_collective=True, strategy=strategy)
        rank_id = dist.get_rank()
        m1 = 1.0
        m2 = 0.5
        m3 = 0.0
        s = 64.0
        batch_size = 2
        feature_length = 4
        num_class_per_card = [4, 8]
        num_classes = paddle.sum(paddle.to_tensor(num_class_per_card))

        label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64')
        label_list = []
        dist.all_gather(label_list, label)
        label = paddle.concat(label_list, axis=0)

        X = paddle.randn(
            shape=[batch_size, feature_length],
            dtype='float64')
        X_list = []
        dist.all_gather(X_list, X)
        X = paddle.concat(X_list, axis=0)
        X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
        X = paddle.divide(X, X_l2)

        W = paddle.randn(
            shape=[feature_length, num_class_per_card[rank_id]],
            dtype='float64')
        W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
        W = paddle.divide(W, W_l2)

        logits = paddle.matmul(X, W)
        loss, softmax = paddle.nn.functional.margin_cross_entropy(
            logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None)

        print(logits)
        print(label)
        print(loss)
        print(softmax)

        # python -m paddle.distributed.launch --gpus=0,1 test_margin_cross_entropy.py
        ## for rank0 input
        #Tensor(shape=[4, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #       [[ 0.32888934,  0.02408748, -0.02763289,  0.18173063],
        #        [-0.52893978, -0.10623845, -0.21596515, -0.06432517],
        #        [-0.00536345, -0.03924667,  0.66735314, -0.28640926],
        #        [-0.09907366, -0.48534973, -0.10365338, -0.39472322]])
        #Tensor(shape=[4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        #       [11, 1 , 10, 11])

        ## for rank1 input
        #Tensor(shape=[4, 8], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
        #       [[ 0.68654754,  0.28137170,  0.69694954, -0.60923933, -0.57077653,  0.54576703, -0.38709028,  0.56028204],
        #        [-0.80360371, -0.03042448, -0.45107338,  0.49559349,  0.69998950, -0.45411693,  0.61927630, -0.82808600],
        #        [ 0.11457570, -0.34785879, -0.68819499, -0.26189226, -0.48241491, -0.67685711,  0.06510185,  0.49660849],
        #        [ 0.31604851,  0.52087884,  0.53124749, -0.86176582, -0.43426329,  0.34786144, -0.10850784,  0.51566383]])
        #Tensor(shape=[4], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
        #       [11, 1 , 10, 11])

        ## for rank0 output
        #Tensor(shape=[4, 1], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #       [[38.96608230],
        #        [81.28152394],
        #        [69.67229865],
        #        [31.74197251]])
        #Tensor(shape=[4, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #       [[0.00000000, 0.00000000, 0.00000000, 0.00000000],
        #        [0.00000000, 0.00000000, 0.00000000, 0.00000000],
        #        [0.00000000, 0.00000000, 0.99998205, 0.00000000],
        #        [0.00000000, 0.00000000, 0.00000000, 0.00000000]])
        ## for rank1 output
        #Tensor(shape=[4, 1], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
        #       [[38.96608230],
        #        [81.28152394],
        #        [69.67229865],
        #        [31.74197251]])
        #Tensor(shape=[4, 8], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
        #       [[0.33943993, 0.00000000, 0.66051859, 0.00000000, 0.00000000, 0.00004148, 0.00000000, 0.00000000],
        #        [0.00000000, 0.00000000, 0.00000000, 0.00000207, 0.99432097, 0.00000000, 0.00567696, 0.00000000],
        #        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00001795],
        #        [0.00000069, 0.33993085, 0.66006319, 0.00000000, 0.00000000, 0.00000528, 0.00000000, 0.00000000]])
    """

    assert reduction in ['mean', 'sum', 'none', None]
    if not (group is False or group is None or hasattr(group, 'is_member')):
        raise ValueError(
            'Expected group is False, None or instance of paddle.distributed.collective.Group \
             (got group: {})'.format(
                group
            )
        )
        return

    if hasattr(group, 'is_member') and not group.is_member():
        return

    ring_id = 0
    rank = 0
    nranks = 1
    if group is not False:
        ring_id = 0 if group is None else group.id
        if core.is_compiled_with_dist():
            parallel_env = paddle.distributed.ParallelEnv()
            global_rank = parallel_env.rank
            rank = (
                global_rank
                if group is None
                else group.get_group_rank(global_rank)
            )
            nranks = parallel_env.world_size if group is None else group.nranks

    input_dims = len(list(logits.shape))
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            'Expected input_dims - 1 = label_dims or input_dims == label_dims\
             (got nput_dims{}, label_dims{})'.format(
                input_dims, label_dims
            )
        )
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=-1)

    if in_dygraph_mode():
        softmax, loss = _C_ops.margin_cross_entropy(
            logits,
            label,
            return_softmax,
            ring_id,
            rank,
            nranks,
            margin1,
            margin2,
            margin3,
            scale,
        )
        if reduction == 'mean':
            loss = paddle.mean(loss)
        elif reduction == 'sum':
            loss = paddle.sum(loss)
        if not return_softmax:
            return loss
        else:
            return loss, softmax
    elif _in_legacy_dygraph():
        softmax, loss = _legacy_C_ops.margin_cross_entropy(
            logits,
            label,
            'ring_id',
            ring_id,
            'rank',
            rank,
            'nranks',
            nranks,
            'margin1',
            margin1,
            'margin2',
            margin2,
            'margin3',
            margin3,
            'scale',
            scale,
            'return_softmax',
            return_softmax,
        )
        if reduction == 'mean':
            loss = paddle.mean(loss)
        elif reduction == 'sum':
            loss = paddle.sum(loss)
        if not return_softmax:
            return loss
        else:
            return loss, softmax

    op_type = 'margin_cross_entropy'
    helper = LayerHelper(op_type, **locals())
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)

    check_variable_and_dtype(
        logits,
        'logits',
        ['float16', 'float32', 'float64'],
        'margin_cross_entropy',
    )
    check_variable_and_dtype(
        label, 'label', ['int32', 'int64'], 'margin_cross_entropy'
    )

    helper.append_op(
        type=op_type,
        inputs={'Logits': logits, 'Label': label},
        outputs={'Softmax': softmax, 'Loss': loss},
        attrs={
            'return_softmax': return_softmax,
            'ring_id': ring_id,
            'rank': rank,
            'nranks': nranks,
            'margin1': margin1,
            'margin2': margin2,
            'margin3': margin3,
            'scale': scale,
        },
    )

    if reduction == 'mean':
        loss = paddle.mean(loss)
    elif reduction == 'sum':
        loss = paddle.sum(loss)

    if not return_softmax:
        return loss
    else:
        return loss, softmax


@deprecated(
    since="2.0.0",
    update_to="paddle.nn.functional.cross_entropy",
    level=1,
    reason=(
        'Please notice that behavior of "paddle.nn.functional.softmax_with_cross_entropy" '
        'and "paddle.nn.functional.cross_entropy" is different.'
    ),
)
def softmax_with_cross_entropy(
    logits,
    label,
    soft_label=False,
    ignore_index=-100,
    numeric_stable_mode=True,
    return_softmax=False,
    axis=-1,
):
    r"""
    This operator implements the cross entropy loss function with softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators
    expects mutually exclusive hard labels, each sample in a batch is in exactly
    one class with a probability of 1.0. Each sample in the batch will have a
    single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::
        \\loss_j=-\text{logits}_{label_j} +\log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::
        \\loss_j= -\sum_{i=0}^{K}\text{label}_i\left(\text{logits}_i - \log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right)\right), j = 1,...,K

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::
        \\max_j&=\max_{i=0}^{K}{\text{logits}_i} \\
                log\_max\_sum_j &= \log\sum_{i=0}^{K}\exp(logits_i - max_j)\\
                softmax_j &= \exp(logits_j - max_j - {log\_max\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64. The input tensor of unscaled log probabilities.
        label (Tensor): The ground truth  ``Tensor`` , data type is the same
            as the ``logits`` . If :attr:`soft_label` is set to :attr:`True`,
            Label is a ``Tensor``  in the same shape with :attr:`logits`.
            If :attr:`soft_label` is set to :attr:`True`, Label is a ``Tensor``
            in the same shape with :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool, optional): A flag to indicate whether to interpretant the given
            labels as soft labels. Default False.
        ignore_index (int, optional): Specifies a target value that is ignored and does
                                      not contribute to the input gradient. Only valid
                                      if :attr:`soft_label` is set to :attr:`False`.
                                      Default: kIgnoreIndex(-100).
        numeric_stable_mode (bool, optional): A flag to indicate whether to use a more
                                              numerically stable algorithm. Only valid
                                              when :attr:`soft_label` is :attr:`False`
                                              and GPU is used. When :attr:`soft_label`
                                              is :attr:`True` or CPU is used, the
                                              algorithm is always numerically stable.
                                              Note that the speed may be slower when use
                                              stable algorithm. Default: True.
        return_softmax (bool, optional): A flag indicating whether to return the softmax
                                         along with the cross entropy loss. Default: False.
        axis (int, optional): The index of dimension to perform softmax calculations. It
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.

    Returns:
        ``Tensor`` or Tuple of two ``Tensor`` : Return the cross entropy loss if \
                                                    `return_softmax` is False, otherwise the tuple \
                                                    (loss, softmax), softmax is in the same shape \
                                                    with input logits and cross entropy loss is in \
                                                    the same shape with input logits except shape \
                                                    in dimension :attr:`axis` as 1.

    Examples:
        .. code-block:: python

            import paddle

            logits = paddle.to_tensor([0.4, 0.6, 0.9], dtype="float32")
            label = paddle.to_tensor([1], dtype="int64")

            out = paddle.nn.functional.softmax_with_cross_entropy(logits=logits, label=label)
            print(out)
            # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1.15328646])
    """
    return fluid_softmax_with_cross_entropy(
        logits,
        label,
        soft_label,
        ignore_index,
        numeric_stable_mode,
        return_softmax,
        axis,
    )


def cross_entropy(
    input,
    label,
    weight=None,
    ignore_index=-100,
    reduction='mean',
    soft_label=False,
    axis=-1,
    use_softmax=True,
    name=None,
):
    r"""

    By default, the cross entropy loss function is implemented using softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable computing.

    Calculate the cross entropy loss function without softmax when use_softmax=False.

    By default, calculate the mean of the result, and you can also affect
    the default behavior by using the reduction parameter. Please refer to the part of
    parameters for details.

    Can be used to calculate the softmax cross entropy loss with soft and hard labels.
    Where, the hard labels mean the actual label value, 0, 1, 2, etc.  And the soft labels
    mean the probability of the actual label, 0.6, 0.8, 0.2, etc.

    The calculation includes the following two steps.

    - **1.softmax cross entropy**

        1. Hard label (each sample can only be assigned into one category)

        1.1. when use_softmax=True

            .. math::
              \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        1.2. when use_softmax=False

            .. math::
              \\loss_j=-\log\left({P}_{label_j}\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).


        2. Soft label (each sample is assigned to multiple categories with a certain probability, and the probability sum is 1).

        2.1. when use_softmax=True

            .. math::
              \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        2.2. when use_softmax=False

            .. math::
              \\loss_j=-\sum_{j=0}^{C}\left({label}_j*\log\left({P}_{label_j}\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).




    - **2. Weight and reduction processing**

        1. Weight

            If the ``weight`` parameter is ``None`` , go to the next step directly.

            If the ``weight`` parameter is not ``None`` , the cross entropy of each sample is weighted by weight
            according to soft_label = False or True as follows.

            1.1. Hard labels (soft_label = False)

            .. math::
                \\loss_j=loss_j*weight[label_j]


            1.2. Soft labels (soft_label = True)

             .. math::
                \\loss_j=loss_j*\sum_{i}\left(weight[label_i]*logits_i\right)

        2. reduction

            2.1 if the ``reduction`` parameter is ``none``

                Return the previous result directly

            2.2 if the ``reduction`` parameter is ``sum``

                Return the sum of the previous results

            .. math::
               \\loss=\sum_{j}loss_j

            2.3 if the ``reduction`` parameter is ``mean`` , it will be processed according to
            the ``weight`` parameter as follows.

            2.3.1. If the  ``weight``  parameter is ``None``

                   Return the average value of the previous results

            .. math::
                \\loss=\sum_{j}loss_j/N

                  where, N is the number of samples and C is the number of categories.

            2.3.2. If the 'weight' parameter is not 'None', the weighted average value of the previous result will be returned

            1. Hard labels (soft_label = False)

            .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}weight[label_j]

            2. Soft labels (soft_label = True)

            .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}\left(\sum_{i}weight[label_i]\right)


    Parameters:
        input (Tensor): the data type is float32, float64. Shape is :math:`[N_1, N_2, ..., N_k, C]`, where C is number of classes, ``k >= 1`` .

            Note:
                1. when use_softmax=True, it expects unscaled logits. This operator should not be used with the output of softmax operator, which will produce incorrect results.
                2. when use_softmax=False, it expects the output of softmax operator.

        label (Tensor):
            1. If soft_label=False, the shape is
            :math:`[N_1, N_2, ..., N_k]` or :math:`[N_1, N_2, ..., N_k, 1]`, k >= 1.
            the data type is int32, int64, float32, float64, where each value is [0, C-1].

            2. If soft_label=True, the shape and data type should be same with ``input`` ,
            and the sum of the labels for each sample should be 1.

        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size C and the data type is float32, float64.
            Default is ``'None'`` .
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the loss. A negative value means that no label
            value needs to be ignored. Only valid when soft_label = False.
            Default is ``-100`` .
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        soft_label (bool, optional): Indicate whether label is soft. Default is ``False``.
        axis (int, optional):The index of dimension to perform softmax calculations.
            It should be in range :math:`[-1, rank - 1]`, where :math:`rank` is the
            number of dimensions of input :attr:`input`.
            Default is ``-1`` .
        use_softmax (bool, optional): Indicate whether compute softmax before cross_entropy.
            Default is ``True``.
        name (str, optional): The name of the operator. Default is ``None`` .
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:

        Tensor. Return the softmax cross_entropy loss of ``input`` and ``label``.
        The data type is the same as input.

        If :attr:`reduction` is ``'mean'`` or ``'sum'`` , the dimension of return value is ``1``.

        If :attr:`reduction` is ``'none'``:

        1. If soft_label = False, the dimension of return value is the same with ``label`` .

        2. if soft_label = True, the dimension of return value is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:
        .. code-block:: python

            # hard labels
            import paddle
            paddle.seed(99999)
            N=100
            C=200
            reduction='mean'
            input =  paddle.rand([N, C], dtype='float64')
            label =  paddle.randint(0, C, shape=[N], dtype='int64')
            weight = paddle.rand([C], dtype='float64')

            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction=reduction)
            dy_ret = cross_entropy_loss(
                                        input,
                                        label)
            print(dy_ret)
            # Tensor(shape=[1], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            #        [5.34043430])

        .. code-block:: python

            # soft labels
            import paddle
            paddle.seed(99999)
            axis = -1
            ignore_index = -100
            N = 4
            C = 3
            shape = [N, C]
            reduction='mean'
            weight = None
            logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            labels /= paddle.sum(labels, axis=axis, keepdim=True)
            paddle_loss_mean = paddle.nn.functional.cross_entropy(
                                                                    logits,
                                                                    labels,
                                                                    soft_label=True,
                                                                    axis=axis,
                                                                    weight=weight,
                                                                    reduction=reduction)
            print(paddle_loss_mean)
            # Tensor(shape=[1], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            #        [1.11043464])

    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in softmax_cross_entropy"
            "should be 'sum', 'mean' or 'none', but received %s, which is not allowed."
            % reduction
        )
    if ignore_index > 0 and soft_label:
        raise ValueError(
            "When soft_label == True, the value of 'ignore_index' in softmax_cross_entropy"
            "should be '-100', but received %s, which is not allowed."
            % ignore_index
        )

    input_dims = len(list(input.shape))
    if input_dims == 0:
        raise ValueError('The dimention of input should be larger than zero!')

    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            'Expected nput_dims - 1 = label_dims or input_dims == label_dims\
             (got nput_dims{}, label_dims{})'.format(
                input_dims, label_dims
            )
        )
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=axis)

    if in_dygraph_mode():
        if not soft_label:
            valid_label = (
                paddle.cast(label != ignore_index, dtype=label.dtype) * label
            )
        if core.is_compiled_with_npu() or core.is_compiled_with_mlu():
            if not soft_label:
                _, _, out = _legacy_C_ops.softmax_with_cross_entropy(
                    input,
                    valid_label,
                    'soft_label',
                    soft_label,
                    'ignore_index',
                    ignore_index,
                    'numeric_stable_mode',
                    True,
                    'axis',
                    axis,
                    'use_softmax',
                    use_softmax,
                )
            else:
                _, _, out = _legacy_C_ops.softmax_with_cross_entropy(
                    input,
                    label,
                    'soft_label',
                    soft_label,
                    'ignore_index',
                    ignore_index,
                    'numeric_stable_mode',
                    True,
                    'axis',
                    axis,
                    'use_softmax',
                    use_softmax,
                )
        else:
            _, out = _C_ops.cross_entropy_with_softmax(
                input, label, soft_label, use_softmax, True, ignore_index, axis
            )

        if weight is not None:

            # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
            if soft_label:
                # chajchaj:
                # weight's shape is C, where C is class num.
                # for 1d case: label's shape is [N,C], weight_gather's shape is N.
                # for 2d case: label's shape is [N,H,W,C], weight_gather's shape is [N,H,W].
                weight_gather = paddle.matmul(
                    x=paddle.cast(label, weight.dtype),
                    y=weight,
                    transpose_x=False,
                    transpose_y=True,
                )
                out_shape = list(out.shape)
                weight_gather_reshape = reshape(weight_gather, shape=out_shape)
                out = paddle.cast(out, weight_gather_reshape.dtype)

                out = _C_ops.multiply(out, weight_gather_reshape)
            else:
                if input.shape[axis] != weight.shape[-1]:
                    raise ValueError(
                        "input's class_dimension({}) must equal to "
                        "weight's class_dimension({}) "
                        "when weight is provided".format(
                            input.shape[axis], weight.shape[-1]
                        )
                    )

                ignore_weight_mask = paddle.cast(
                    (label != ignore_index), out.dtype
                )
                if (
                    ignore_weight_mask.ndim > 1
                    and ignore_weight_mask.shape[axis] == 1
                ):
                    # TODO: Temporarily use squeeze instead of squeeze_
                    ignore_weight_mask = paddle.squeeze(
                        ignore_weight_mask, axis
                    )
                if axis != -1 and axis != valid_label.ndim - 1:
                    temp_perm = (
                        list(range(axis % valid_label.ndim))
                        + list(
                            range(
                                (axis % valid_label.ndim + 1), valid_label.ndim
                            )
                        )
                        + [axis % valid_label.ndim]
                    )
                    weight_gather = _C_ops.gather_nd(
                        weight, valid_label.transpose(temp_perm)
                    )
                else:
                    weight_gather = _C_ops.gather_nd(weight, valid_label)
                weight_gather = _C_ops.multiply(
                    weight_gather, ignore_weight_mask
                )
                input_shape = list(label.shape)
                weight_gather_reshape = reshape(
                    weight_gather, shape=input_shape
                )
                out = paddle.cast(out, weight_gather_reshape.dtype)
                out = _C_ops.multiply(out, weight_gather_reshape)

        if reduction == "sum":
            #   because of fluid_softmax_with_cross_entropy op's inner logic,
            #   in the out tensor of this op, the loss of sample with class_index==ignore_index is 0
            #   so, reduce_sum all directly is ok
            return _C_ops.sum(out, [], None, False)
        elif reduction == "mean":
            # 1. if weight==none,
            #     numerator: reduce_sum all loss directly is ok causeof fluid_softmax_with_cross_entropy's inner logic
            #     denominator: count sample num with class_index!=ignore_index
            # 2. else
            #     numerator: loss's weighted sum
            #     denominator: cal the sum of weight where the sample's class_index!=ignore_index
            if ignore_index >= 0:
                out_sum = _C_ops.sum(out, [], None, False)
                # for each label[i],set 1 or 0, according to ignore_index
                # mask[i]=0, if label[i]==ignore_index
                # mask[i]=1, otherwise
                mask = label != ignore_index
                if weight is None:
                    mask = paddle.cast(mask, dtype=out_sum.dtype)
                    count = _C_ops.sum(mask, [], None, False)
                    ret = out_sum / (count + (count == 0.0))
                else:
                    mask = paddle.cast(mask, weight_gather_reshape.dtype)
                    weight_ignored = _C_ops.multiply(
                        mask, weight_gather_reshape
                    )
                    weight_sum = _C_ops.sum(weight_ignored, [], None, False)
                    ret = out_sum / (weight_sum + (weight_sum == 0.0))
                return ret
            elif weight is not None:
                out_sum = _C_ops.sum(out, [], None, False)
                total_weight = _C_ops.sum(
                    weight_gather_reshape, [], None, False
                )
                return out_sum / (total_weight + (total_weight == 0.0))
            else:
                return _C_ops.mean_all(out)

        else:
            if input_dims - 1 == label_dims:
                out = paddle.squeeze(out, axis=axis)
            return out

    elif _in_legacy_dygraph():
        if not soft_label:
            valid_label = (
                paddle.cast(label != ignore_index, dtype=label.dtype) * label
            )
            label_min = paddle.min(valid_label)
            label_max = paddle.max(valid_label)
            if label_min < 0:
                raise ValueError(
                    "Target {} is out of lower bound.".format(label_min.item())
                )
            if label_max >= input.shape[axis]:
                raise ValueError(
                    "Target {} is out of upper bound.".format(label_max.item())
                )
        if core.is_compiled_with_npu() or core.is_compiled_with_mlu():
            if not soft_label:
                _, _, out = _legacy_C_ops.softmax_with_cross_entropy(
                    input,
                    valid_label,
                    'soft_label',
                    soft_label,
                    'ignore_index',
                    ignore_index,
                    'numeric_stable_mode',
                    True,
                    'axis',
                    axis,
                    'use_softmax',
                    use_softmax,
                )
            else:
                _, _, out = _legacy_C_ops.softmax_with_cross_entropy(
                    input,
                    label,
                    'soft_label',
                    soft_label,
                    'ignore_index',
                    ignore_index,
                    'numeric_stable_mode',
                    True,
                    'axis',
                    axis,
                    'use_softmax',
                    use_softmax,
                )
        else:
            _, out = _legacy_C_ops.softmax_with_cross_entropy(
                input,
                label,
                'soft_label',
                soft_label,
                'ignore_index',
                ignore_index,
                'numeric_stable_mode',
                True,
                'axis',
                axis,
                'use_softmax',
                use_softmax,
            )

        if weight is not None:

            # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
            if soft_label:
                # chajchaj:
                # weight's shape is C, where C is class num.
                # for 1d case: label's shape is [N,C], weight_gather's shape is N.
                # for 2d case: label's shape is [N,H,W,C], weight_gather's shape is [N,H,W].
                weight_gather = paddle.matmul(
                    x=paddle.cast(label, weight.dtype),
                    y=weight,
                    transpose_x=False,
                    transpose_y=True,
                )
                out_shape = list(out.shape)
                weight_gather_reshape = reshape(weight_gather, shape=out_shape)
                out = paddle.cast(out, weight_gather_reshape.dtype)

                out = _legacy_C_ops.elementwise_mul(out, weight_gather_reshape)

            else:
                if input.shape[axis] != weight.shape[-1]:
                    raise ValueError(
                        "input's class_dimension({}) must equal to "
                        "weight's class_dimension({}) "
                        "when weight is provided".format(
                            input.shape[axis], weight.shape[-1]
                        )
                    )

                ignore_weight_mask = paddle.cast(
                    (label != ignore_index), out.dtype
                )
                if (
                    ignore_weight_mask.ndim > 1
                    and ignore_weight_mask.shape[axis] == 1
                ):
                    # TODO: Temporarily use squeeze instead of squeeze_
                    ignore_weight_mask = paddle.squeeze(
                        ignore_weight_mask, axis
                    )
                if axis != -1 and axis != valid_label.ndim - 1:
                    temp_perm = (
                        list(range(axis % valid_label.ndim))
                        + list(
                            range(
                                (axis % valid_label.ndim + 1), valid_label.ndim
                            )
                        )
                        + [axis % valid_label.ndim]
                    )
                    weight_gather = _legacy_C_ops.gather_nd(
                        weight, valid_label.transpose(temp_perm)
                    )
                else:
                    weight_gather = _legacy_C_ops.gather_nd(weight, valid_label)
                weight_gather = _legacy_C_ops.elementwise_mul(
                    weight_gather, ignore_weight_mask
                )
                input_shape = list(label.shape)
                weight_gather_reshape = reshape(
                    weight_gather, shape=input_shape
                )
                out = paddle.cast(out, weight_gather_reshape.dtype)
                out = _legacy_C_ops.elementwise_mul(out, weight_gather_reshape)

        if reduction == "sum":
            #   because of fluid_softmax_with_cross_entropy op's inner logic,
            #   in the out tensor of this op, the loss of sample with class_index==ignore_index is 0
            #   so, reduce_sum all directly is ok
            return _legacy_C_ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == "mean":
            # 1. if weight==none,
            #     numerator: reduce_sum all loss directly is ok causeof fluid_softmax_with_cross_entropy's inner logic
            #     denominator: count sample num with class_index!=ignore_index
            # 2. else
            #     numerator: loss's weighted sum
            #     denominator: cal the sum of weight where the sample's class_index!=ignore_index
            if ignore_index >= 0:
                out_sum = _legacy_C_ops.reduce_sum(out, 'reduce_all', True)
                # for each label[i],set 1 or 0, according to ignore_index
                # mask[i]=0, if label[i]==ignore_index
                # mask[i]=1, otherwise
                mask = label != ignore_index
                if weight is None:
                    mask = paddle.cast(mask, dtype=out_sum.dtype)
                    count = _legacy_C_ops.reduce_sum(mask, 'reduce_all', True)
                    ret = out_sum / (count + (count == 0.0))
                else:
                    mask = paddle.cast(mask, weight_gather_reshape.dtype)
                    weight_ignored = _legacy_C_ops.elementwise_mul(
                        mask, weight_gather_reshape
                    )
                    weight_sum = _legacy_C_ops.reduce_sum(
                        weight_ignored, 'reduce_all', True
                    )
                    ret = out_sum / (weight_sum + (weight_sum == 0.0))
                return ret
            elif weight is not None:
                out_sum = _legacy_C_ops.reduce_sum(out, 'reduce_all', True)
                total_weight = _legacy_C_ops.reduce_sum(
                    weight_gather_reshape, 'reduce_all', True
                )
                return out_sum / (total_weight + (total_weight == 0.0))
            else:
                return _legacy_C_ops.mean(out)
        else:
            if input_dims - 1 == label_dims:
                out = paddle.squeeze(out, axis=axis)
            return out

    check_variable_and_dtype(
        input,
        'input',
        ['float16', 'float32', 'float64'],
        'softmax_cross_entropy',
    )
    check_variable_and_dtype(
        label,
        'label',
        ['uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'],
        'softmax_cross_entropy',
    )
    attrs = {
        'soft_label': soft_label,
        'ignore_index': ignore_index,
        'numeric_stable_mode': True,
        'axis': axis,
        'use_softmax': use_softmax,
    }
    helper = LayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_variable_for_type_inference(dtype=input.dtype)
    out = helper.create_variable_for_type_inference(dtype=input.dtype)

    outputs = {'Softmax': softmax, 'Loss': out}
    if core.is_compiled_with_npu() or core.is_compiled_with_mlu():
        backprop = helper.create_variable_for_type_inference(dtype=input.dtype)
        outputs['Backprop'] = backprop
    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': input, 'Label': label},
        outputs=outputs,
        attrs=attrs,
    )

    if weight is not None:
        check_variable_and_dtype(
            weight, 'weight', ['float32', 'float64'], 'softmax_cross_entropy'
        )
        weight_name = name if reduction == 'none' else None
        if soft_label:
            # chajchaj:
            # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
            # weight's shape is C, where C is class num.
            # for 1d case: label's shape is [N,C], weight_gather's shape is N.
            # for 2d case: label's shape is [N,H,W,C], weight_gather's shape is [N,H,W].
            weight_gather = paddle.matmul(
                x=paddle.cast(label, weight.dtype),
                y=weight,
                transpose_x=False,
                transpose_y=True,
            )

            out_shape = list(out.shape)
            weight_gather_reshape = reshape(weight_gather, shape=out_shape)
            out = paddle.cast(out, weight_gather_reshape.dtype)
        else:
            if input.shape[axis] != weight.shape[-1]:
                raise ValueError(
                    "input's class_dimension({}) must equal to "
                    "weight's class_dimension({}) "
                    "when weight is provided".format(
                        input.shape[axis], weight.shape[-1]
                    )
                )

            valid_label = paddle.multiply(
                paddle.cast(label != ignore_index, dtype=label.dtype), label
            )
            ignore_weight_mask = paddle.cast(
                (label != ignore_index), input.dtype
            )
            if (
                ignore_weight_mask.ndim > 1
                and ignore_weight_mask.shape[axis] == 1
            ):
                ignore_weight_mask = paddle.squeeze(ignore_weight_mask, axis)
            if axis != -1 and axis != valid_label.ndim - 1:
                temp_perm = (
                    list(range(axis % valid_label.ndim))
                    + list(
                        range((axis % valid_label.ndim + 1), valid_label.ndim)
                    )
                    + [axis % valid_label.ndim]
                )
                weight_gather = paddle.gather_nd(
                    weight, paddle.transpose(valid_label, temp_perm)
                )
            else:
                weight_gather = paddle.gather_nd(weight, valid_label)
            weight_gather = paddle.multiply(weight_gather, ignore_weight_mask)

            input_shape = list(label.shape)
            weight_gather_reshape = reshape(weight_gather, shape=input_shape)
        out = paddle.multiply(out, weight_gather_reshape, name=weight_name)

    if reduction == "sum":
        return paddle.sum(out, name=name)
    elif reduction == "mean":
        if ignore_index >= 0:
            out_sum = paddle.sum(out, name=name)
            # for each label[i],set 1 or 0, according to ignore_index
            # mask[i]=0, if label[i]==ignore_index
            # mask[i]=1, otherwise
            mask = label != ignore_index
            if weight is None:
                mask = paddle.cast(mask, dtype=out_sum.dtype)
                count = paddle.sum(mask, name=name)
                ret = out_sum / (count + (count == 0.0))
            else:
                mask = paddle.cast(mask, weight_gather_reshape.dtype)
                weight_ignored = paddle.multiply(mask, weight_gather_reshape)
                weight_sum = paddle.sum(weight_ignored, name=name)
                ret = out_sum / (weight_sum + (weight_sum == 0.0))
            return ret
        elif weight is not None:
            out_sum = paddle.sum(out, name=name)
            total_weight = paddle.sum(weight_gather_reshape)
            return out_sum / (total_weight + (total_weight == 0.0))
        else:
            return paddle.mean(out, name=name)

    else:
        if input_dims - 1 == label_dims:
            out = paddle.squeeze(out, axis=axis)

        return out


def sigmoid_focal_loss(
    logit,
    label,
    normalizer=None,
    alpha=0.25,
    gamma=2.0,
    reduction='sum',
    name=None,
):
    r"""
    `Focal Loss <https://arxiv.org/abs/1708.02002>`_ is proposed to address the
    foreground-background class imbalance for classification tasks. It down-weights
    easily-classified examples and thus focuses training on hard examples. For example,
    it is used in one-stage object detection where the foreground-background class
    imbalance is extremely high.

    This operator measures focal loss function as follows:

    .. math::
           Out = -Labels * alpha * {(1 - \sigma(Logit))}^{gamma}\log(\sigma(Logit)) - (1 - Labels) * (1 - alpha) * {\sigma(Logit)}^{gamma}\log(1 - \sigma(Logit))

    We know that :math:`\sigma(Logit) = \frac{1}{1 + \exp(-Logit)}`.

    Then, if :attr:`normalizer` is not None, this operator divides the
    normalizer tensor on the loss `Out`:

    .. math::
           Out = \frac{Out}{normalizer}

    Finally, this operator applies reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, the operator will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target ``label`` is 0 for the negative class and is 1 for the positive class.

    Args:
        logit (Tensor): The input logit tensor. The shape is [N, *], where N is batch_size,
            `*` means any number of additional dimensions. The ``logit`` is usually the
            output of a convolution layer. Available dtype is float32, float64.
        label (Tensor): The target label tensor with the same shape as
            ``logit``. The target label whose value should be numbers between 0 and 1.
            Available dtype is float32, float64.
        normalizer (Tensor, optional): The number normalizes the focal loss. It has to be
            a 1-D Tensor whose shape is `[1, ]`. The data type is float32, float64.
            For object detection task, it is the number of positive samples.
            If set to None, the focal loss will not be normalized. Default is None.
        alpha(int|float, optional): Hyper-parameter to balance the positive and negative example,
            it should be between 0 and 1.  Default value is set to 0.25.
        gamma(int|float, optional): Hyper-parameter to modulate the easy and hard examples.
            Default value is set to 2.0.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'sum'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[1]`, otherwise the shape is the same as ``logit``. The same dtype as ``logit`` tensor.

    Examples:

        .. code-block:: python

            import paddle

            logit = paddle.to_tensor([[0.97, 0.91, 0.03], [0.55, 0.43, 0.71]], dtype='float32')
            label = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32')
            one = paddle.to_tensor([1.], dtype='float32')
            fg_label = paddle.greater_equal(label, one)
            fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32'))
            output = paddle.nn.functional.sigmoid_focal_loss(logit, label, normalizer=fg_num)
            print(output)  # [0.65782464]

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in sigmoid_focal_loss "
            "should be 'sum', 'mean' or 'none', but received %s, which is not allowed."
            % reduction
        )

    if normalizer is not None:
        check_variable_and_dtype(
            normalizer,
            'normalizer',
            ['float32', 'float64'],
            'sigmoid_focal_loss',
        )
        normalizer_shape = list(normalizer.shape)
        normalizer_dims = len(normalizer_shape)
        if normalizer_dims > 1:
            raise ValueError(
                "Expected one dimension of normalizer in sigmoid_focal_loss but got {}.".format(
                    normalizer_dims
                )
            )

    if in_dygraph_mode():
        place = _current_expected_place()
        one = _C_ops.full(logit.shape, float(1.0), logit.dtype, place)

        loss = _C_ops.sigmoid_cross_entropy_with_logits(
            logit, label, False, -100
        )

        pred = _C_ops.sigmoid(logit)

        p_t = _C_ops.add(
            _C_ops.multiply(pred, label),
            _C_ops.multiply(
                _C_ops.subtract(one, pred), _C_ops.subtract(one, label)
            ),
        )

        alpha = fluid.dygraph.base.to_variable([alpha], dtype=loss.dtype)
        alpha_t = _C_ops.add(
            _C_ops.multiply(alpha, label),
            _C_ops.multiply(
                _C_ops.subtract(one, alpha), _C_ops.subtract(one, label)
            ),
        )
        loss = _C_ops.multiply(alpha_t, loss)

        gamma = fluid.dygraph.base.to_variable([gamma], dtype=loss.dtype)
        gamma_t = _C_ops.pow(_C_ops.subtract(one, p_t), gamma)
        loss = _C_ops.multiply(gamma_t, loss)

        if normalizer is not None:
            loss = _C_ops.divide(loss, normalizer)

        if reduction == "sum":
            return _C_ops.sum(loss, [], None, False)
        elif reduction == "mean":
            return _C_ops.mean_all(loss)

        return loss

    elif _in_legacy_dygraph():
        one = _varbase_creator(dtype=logit.dtype)
        _legacy_C_ops.fill_constant(
            one,
            'value',
            float(1.0),
            'force_cpu',
            False,
            'dtype',
            one.dtype,
            'str_value',
            '1.0',
            'shape',
            logit.shape,
        )
        loss = _legacy_C_ops.sigmoid_cross_entropy_with_logits(logit, label)

        pred = _legacy_C_ops.sigmoid(logit)

        p_t = _legacy_C_ops.elementwise_add(
            _legacy_C_ops.elementwise_mul(pred, label),
            _legacy_C_ops.elementwise_mul(
                _legacy_C_ops.elementwise_sub(one, pred),
                _legacy_C_ops.elementwise_sub(one, label),
            ),
        )

        alpha = fluid.dygraph.base.to_variable([alpha], dtype=loss.dtype)
        alpha_t = _legacy_C_ops.elementwise_add(
            _legacy_C_ops.elementwise_mul(alpha, label),
            _legacy_C_ops.elementwise_mul(
                _legacy_C_ops.elementwise_sub(one, alpha),
                _legacy_C_ops.elementwise_sub(one, label),
            ),
        )
        loss = _legacy_C_ops.elementwise_mul(alpha_t, loss)

        gamma = fluid.dygraph.base.to_variable([gamma], dtype=loss.dtype)
        gamma_t = _legacy_C_ops.elementwise_pow(
            _legacy_C_ops.elementwise_sub(one, p_t), gamma
        )
        loss = _legacy_C_ops.elementwise_mul(gamma_t, loss)

        if normalizer is not None:
            loss = _legacy_C_ops.elementwise_div(loss, normalizer)

        if reduction == "sum":
            return _legacy_C_ops.reduce_sum(loss, 'reduce_all', True)
        elif reduction == "mean":
            return _legacy_C_ops.mean(loss)

        return loss

    check_variable_and_dtype(
        logit, 'logit', ['float32', 'float64'], 'sigmoid_focal_loss'
    )
    check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'sigmoid_focal_loss'
    )

    bce_name = None
    if reduction == 'none' and normalizer is None:
        bce_name = name
    loss = paddle.nn.functional.binary_cross_entropy_with_logits(
        logit, label, reduction='none', name=bce_name
    )

    pred = paddle.nn.functional.sigmoid(logit)
    p_t = pred * label + (1 - pred) * (1 - label)

    alpha_t = alpha * label + (1 - alpha) * (1 - label)
    loss = paddle.multiply(alpha_t, loss)

    gamma_t = paddle.pow((1 - p_t), gamma)
    loss = paddle.multiply(gamma_t, loss)

    if normalizer is not None:
        normalizer_name = name if reduction == 'none' else None
        loss = paddle.divide(loss, normalizer, name=normalizer_name)

    if reduction == 'mean':
        loss = paddle.mean(loss, name=name)
    elif reduction == 'sum':
        loss = paddle.sum(loss, name=name)

    return loss


def multi_label_soft_margin_loss(
    input, label, weight=None, reduction="mean", name=None
):
    r"""
    Calculate a multi-class multi-classification
    hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
    and output :math:`y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`, \
    :math:`y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}`, \
    :math:`0 \leq y[j] \leq \text{x.size}(0)-1`, \
    and :math:`i \neq y[j]` for all :math:`i` and :math:`j`.
    :math:`y` and :math:`x` must have the same size.

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes, and if shape is more than 2D, this is (N, C, D1, D2,..., Dk), k >= 1.
        label (Tensor): Label tensor, the data type is float32 or float64. The shape of label is the same as the shape of input.
        weight (Tensor,optional): a manual rescaling weight given to each class.
                If given, has to be a Tensor of size C and the data type is float32, float64.
                Default is ``'None'`` .
        reduction (str, optional): Indicate how to average the loss by batch_size,
                the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``
        name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means number of classes, available dtype is float32, float64. The sum operationoperates over all the elements.
        label: N-D Tensor, same shape as the input.
        weight:N-D Tensor, the shape is [N,1]
        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

    Returns:
        Tensor, The tensor variable storing the multi_label_soft_margin_loss of input and label.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            # label elements in {1., -1.}
            label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)
            loss = F.multi_label_soft_margin_loss(input, label, reduction='none')
            print(loss)
            # Tensor([3.49625897, 0.71111226, 0.43989015])
            loss = F.multi_label_soft_margin_loss(input, label, reduction='mean')
            print(loss)
            # Tensor([1.54908717])
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'multi_label_soft_margin_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction)
        )

    if not (input.shape == label.shape):
        raise ValueError(
            "The input and label should have same dimension,"
            "but received {}!={}".format(input.shape, label.shape)
        )

    if not _non_static_mode():
        check_variable_and_dtype(
            input,
            'input',
            ['float32', 'float64'],
            'multilabel_soft_margin_loss',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['float32', 'float64'],
            'multilabel_soft_margin_loss',
        )

    loss = -(
        label * paddle.nn.functional.log_sigmoid(input)
        + (1 - label) * paddle.nn.functional.log_sigmoid(-input)
    )

    if weight is not None:
        if not _non_static_mode():
            check_variable_and_dtype(
                weight,
                'weight',
                ['float32', 'float64'],
                'multilabel_soft_margin_loss',
            )
        loss = loss * weight

    loss = loss.mean(axis=-1)  # only return N loss values

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return paddle.mean(loss)
    elif reduction == "sum":
        return paddle.sum(loss)


def hinge_embedding_loss(input, label, margin=1.0, reduction='mean', name=None):
    r"""
    Calculates hinge_embedding_loss. Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`(containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the L1 pairwise distance as :math:`x`,
    and is typically used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64.
            the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.
        label (Tensor): Label tensor containing 1 or -1, the data type is float32 or float64.
            The shape of label is the same as the shape of input.
        margin (float, optional): Specifies the hyperparameter margin to be used.
            The value determines how large the input need to be to calculate in
            hinge_embedding_loss. When label is -1, Input smaller than margin are minimized with hinge_embedding_loss.
            Default = 1.0
        reduction (str, optional): Indicate how to average the loss by batch_size.
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:

        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64. The sum operationoperates over all the elements.

        label: N-D Tensor, same shape as the input. tensor elements should containing 1 or -1, the data type is float32 or float64.

        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

    Returns:
        Tensor. The tensor variable storing the hinge_embedding_loss of input and label.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            # label elements in {1., -1.}
            label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

            loss = F.hinge_embedding_loss(input, label, margin=1.0, reduction='none')
            print(loss)
            # Tensor([[0., -2., 0.],
            #         [0., -1., 2.],
            #         [1., 1., 1.]])

            loss = F.hinge_embedding_loss(input, label, margin=1.0, reduction='mean')
            print(loss)
            # Tensor([0.22222222])
    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'hinge_embedding_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction)
        )

    if not _non_static_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'hinge_embedding_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'hinge_embedding_loss'
        )

    zero_ = paddle.zeros([1], dtype=input.dtype)
    loss = paddle.where(label == 1.0, input, zero_) + paddle.where(
        label == -1.0, paddle.nn.functional.relu(margin - input), zero_
    )

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def cosine_embedding_loss(
    input1, input2, label, margin=0, reduction='mean', name=None
):
    r"""
    Compute the cosine embedding loss of Tensor ``input1``, ``input2`` and ``label`` as follows.

    If label = 1, then the loss value can be calculated as follow:

    .. math::
        Out = 1 - cos(input1, input2)

    If label = -1, then the loss value can be calculated as follow:

    .. math::
        Out = max(0, cos(input1, input2)) - margin

    The operator cos can be described as follow:
     .. math::
        cos(x1, x2) = \frac{x1 \cdot{} x2}{\Vert x1 \Vert_2 * \Vert x2 \Vert_2}

    Parameters:
        input1 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        input2 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        label (Tensor): tensor with shape: [N] or [1], 'N' means the length of input array. The target labels values should be -1 or 1.
                         Available dtypes are int32, int64, float32, float64.
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
                         :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
                         default value is :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
                         ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                         ``'mean'``: the sum of the output will be divided by the number of elements in the output
                         ``'sum'``: the output will be summed.
        name (str, optional): Name for the operation (optional, default is None).
                         For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the cosine embedding Loss of Tensor ``input1`` ``input2`` and ``label``.
            If `reduction` is ``'none'``, the shape of output loss is [N], the same as ``input`` .
            If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1].

    Examples:
        .. code-block:: python

            import paddle

            input1 = paddle.to_tensor([[1.6, 1.2, -0.5], [3.2, 2.6, -5.8]], 'float32')
            input2 = paddle.to_tensor([[0.5, 0.5, -1.8], [2.3, -1.4, 1.1]], 'float32')
            label = paddle.to_tensor([1, -1], 'int64')

            output = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
            print(output)  # [0.21155193]

            output = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='sum')
            print(output)  # [0.42310387]

            output = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='none')
            print(output)  # [0.42310387, 0.        ]

    """
    if len(label.shape) != 1:
        raise ValueError(
            "1D target tensor expected, multi-target not supported"
        )

    if input1.shape != input2.shape:
        raise ValueError(
            "the shape of input tensor 1 should be equal to input tensor 2, but found inputs with "
            "different sizes"
        )

    if len(input1.shape) > 2:
        raise ValueError(
            "1D target tensor expects 1D or 2D input tensors, but found inputs with different sizes"
        )

    if input1.dtype not in [paddle.float32, paddle.float64]:
        raise ValueError(
            "The data type of input Variable must be 'float32' or 'float64'"
        )
    if label.dtype not in [
        paddle.int32,
        paddle.int64,
        paddle.float32,
        paddle.float64,
    ]:
        raise ValueError(
            "The data type of label Variable must be 'int32', 'int64', 'float32', 'float64'"
        )

    prod_sum = (input1 * input2).sum(axis=-1)
    mag_square1 = paddle.square(input1).sum(axis=-1) + 10e-12
    mag_square2 = paddle.square(input2).sum(axis=-1) + 10e-12
    denom = paddle.sqrt(mag_square1 * mag_square2)
    cos = prod_sum / denom
    zeros = paddle.zeros_like(cos)
    pos = 1 - cos
    neg = paddle.clip(cos - margin, min=0)
    out_pos = paddle.where(label == 1, pos, zeros)
    out_neg = paddle.where(label == -1, neg, zeros)
    out = out_pos + out_neg

    if reduction == 'none':
        return out
    if reduction == 'mean':
        return paddle.mean(out, name=name)
    elif reduction == 'sum':
        return paddle.sum(out, name=name)


def triplet_margin_with_distance_loss(
    input,
    positive,
    negative,
    distance_function=None,
    margin=1.0,
    swap=False,
    reduction='mean',
    name=None,
):
    r"""
    Measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


    where the default distance function

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    or user can defined their own distance functions. `margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to be 0. If `swap` is true, it will compare distance of (input, negative) with
    distance of (negative, positive) and change it to the smaller one. For more details see http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf.

    Parameters:

        input (Tensor):Input tensor, the data type is float32 or float64.
            the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        positive (Tensor):Positive tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        negative (Tensor):Negative tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        distance_function (callable, optional): Quantifies the distance between two tensors. if not specified, 2 norm functions will be used.

        margin (float, optional): A nonnegative margin representing the minimum difference
            between the positive and negative distances required for the loss to be 0. Default value is :math:`1`.

        swap (bool, optional):The distance swap changes the negative distance to the swap distance (distance between positive samples
                and negative samples) if swap distance smaller than negative distance. Default: ``False``.

        reduction (str, optional):Indicate how to average the loss by batch_size.
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output: Tensor. The tensor variable storing the triplet_margin_with_distance_loss of input and positive and negative.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            loss = F.triplet_margin_with_distance_loss(input, positive, negative, margin=1.0, reduction='none')
            print(loss)
            # Tensor([0.        , 0.57496738, 0.        ])


            loss = F.triplet_margin_with_distance_loss(input, positive, negative, margin=1.0, reduction='mean')
            print(loss)
            # Tensor([0.19165580])

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'triplet_margin_with_distance_loss' "
            "should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction)
        )
    if margin < 0:
        raise ValueError(
            "The margin between positive samples and negative samples should be greater than 0."
        )
    if not _non_static_mode():
        check_variable_and_dtype(
            input,
            'input',
            ['float32', 'float64'],
            'triplet_margin_with_distance_loss',
        )
        check_variable_and_dtype(
            positive,
            'positive',
            ['float32', 'float64'],
            'triplet_margin_with_distance_loss',
        )
        check_variable_and_dtype(
            negative,
            'negative',
            ['float32', 'float64'],
            'triplet_margin_with_distance_loss',
        )

    if not (input.shape == positive.shape == negative.shape):
        raise ValueError(
            "input's shape must equal to "
            "positive's shape and  "
            "negative's shape"
        )

    distance_function = (
        distance_function
        if distance_function is not None
        else paddle.nn.PairwiseDistance(2)
    )

    positive_dist = distance_function(input, positive)
    negative_dist = distance_function(input, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = paddle.minimum(negative_dist, swap_dist)

    if not paddle.all(positive_dist > 0) or not paddle.all(negative_dist > 0):
        raise ValueError(
            "The positive distance or negative distance should be greater than 0, "
            "The distance functions should be checked."
        )

    loss = paddle.clip(positive_dist - negative_dist + margin, min=0.0)

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def triplet_margin_loss(
    input,
    positive,
    negative,
    margin=1.0,
    p=2,
    epsilon=1e-6,
    swap=False,
    reduction='mean',
    name=None,
):
    r"""
        Measures the triplet loss given an input
        tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
        This is used for measuring a relative similarity between samples. A triplet
        is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
        examples` respectively). The shapes of all input tensors should be
        :math:`(N, *)`.

        The loss function for each sample in the mini-batch is:

        .. math::
            L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


        where

        .. math::
            d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64.
            the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        positive (Tensor): Positive tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        negative (Tensor): Negative tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        margin (float, Optional): Default: :math:`1`.

        p (int, Optional): The norm degree for pairwise distance. Default: :math:`2`.

        epsilon (float, Optional): Add small value to avoid division by zero,
            default value is 1e-6.

        swap (bool,Optional): The distance swap change the negative distance to the distance between
            positive sample and negative sample. For more details, see `Learning shallow convolutional feature descriptors with triplet losses`.
            Default: ``False``.


        reduction (str, Optional):Indicate how to average the loss by batch_size.
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``

        name (str, Optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output: Tensor. The tensor variable storing the triplet_margin_loss of input and positive and negative.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            loss = F.triplet_margin_loss(input, positive, negative, margin=1.0, reduction='none')
            print(loss)
            # Tensor([0.        , 0.57496738, 0.        ])


            loss = F.triplet_margin_loss(input, positive, negative, margin=1.0, reduction='mean')
            print(loss)
            # Tensor([0.19165580])

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'triplet_margin_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction)
        )
    if margin < 0:
        raise ValueError(
            "The margin between positive samples and negative samples should be greater than 0."
        )
    if not _non_static_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'triplet_margin_loss'
        )
        check_variable_and_dtype(
            positive, 'positive', ['float32', 'float64'], 'triplet_margin_loss'
        )
        check_variable_and_dtype(
            negative, 'negative', ['float32', 'float64'], 'triplet_margin_loss'
        )

    if not (input.shape == positive.shape == negative.shape):
        raise ValueError(
            "input's shape must equal to "
            "positive's shape and  "
            "negative's shape"
        )

    distance_function = paddle.nn.PairwiseDistance(p, epsilon=epsilon)
    positive_dist = distance_function(input, positive)
    negative_dist = distance_function(input, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = paddle.minimum(negative_dist, swap_dist)

    loss = paddle.clip(positive_dist - negative_dist + margin, min=0.0)

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def multi_margin_loss(
    input,
    label,
    p: int = 1,
    margin: float = 1.0,
    weight=None,
    reduction='mean',
    name=None,
):
    r"""
        Measures a multi-class classification hinge loss between input :math:`input` and label :math:`label`:

        For i-th mini-batch sample, the loss in terms of the 1D input :math:`input_i` and scalar
        output :math:`label_i` is:

        .. math::
            \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, \text{margin} - input_i[label_i] + input_i[j])^p}{\text{C}}

        where :math:`0 \leq j \leq \text{C}-1`, :math:`0 \leq i \leq \text{N}-1` and :math:`j \neq label_i`.

        Optionally, you can give non-equal weighting on the classes by passing
        a 1D :attr:`weight` tensor into the constructor.

        The loss function for i-th sample then becomes:

        .. math::
            \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, weight[label_i] * (\text{margin} - input_i[label_i] + input_i[j]))^p}{\text{C}}


    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes.

        label (Tensor): Label tensor, the data type is int32 or int64. The shape of label is (N,)

        p (int, Optional): The power num. Default: :math:`1`.

        margin (float, Optional): Default: :math:`1`.

        weight (Tensor,optional): a manual rescaling weight given to each class.
                If given, has to be a Tensor of shape (C,) and the data type is float32, float64.
                Default is ``'None'`` .


        reduction (str, Optional):Indicate how to calculate the loss by batch_size.
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``

        name (str, Optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output: Tensor. The tensor variable storing the multi_margin_loss of input and label.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            label = paddle.to_tensor([1, 2, 1], dtype=paddle.int32)
            loss = F.multi_margin_loss(input, label, margin=1.0, reduction='none')
            print(loss)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'multi_margin_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction)
        )

    if not _non_static_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'multi_margin_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['int32', 'int64'], 'multi_margin_loss'
        )
    if not (input.shape[0] == label.shape[0]):
        raise ValueError(
            "The label's shape[0] should be equal to input's shape[0], "
            "but received input's shape[0] {} and label's shape[0]:{}. ".format(
                input.shape[0], label.shape[0]
            )
        )
    label = label.reshape((-1, 1))
    index_sample = paddle.index_sample(input, label)
    if weight is not None:
        if not _non_static_mode():
            check_variable_and_dtype(
                weight, 'weight', ['float32', 'float64'], 'multi_margin_loss'
            )
        if not (input.shape[1] == weight.shape[0]):
            raise ValueError(
                "The weight's shape[0] should be equal to input's shape[1]"
                "but received weight's shape[0]: {} and input's shape[1]: {}".format(
                    weight.shape[0], input.shape[1]
                )
            )
        weight = paddle.gather(weight, label, axis=0).reshape((-1, 1))
        loss = paddle.mean(
            paddle.pow(
                paddle.clip(weight * (margin - index_sample + input), min=0.0),
                p,
            ),
            axis=1,
        ) - weight * (margin**p / paddle.shape(input)[1])
    else:
        loss = (
            paddle.mean(
                paddle.pow(
                    paddle.clip(margin - index_sample + input, min=0.0), p
                ),
                axis=1,
            )
            - margin**p / paddle.shape(input)[1]
        )

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def soft_margin_loss(input, label, reduction='mean', name=None):
    """

    The API measures the soft margin loss between input predictions ``input``
    and target labels ``label`` . It can be described as:

    .. math::
        Out = log(1 + exp((-label * input)))

    Parameters:

        input (Tensor): The input predications tensor with shape: ``[N, *]``,
            N is batch_size, `*` means any number of additional dimensions. The ``input`` ranges from -inf to inf.
            Available dtype is float32, float64.

        label (Tensor): The target labels tensor with the same shape as
            ``input``. The target labels which values should be numbers -1 or 1.
            Available dtype is int32, int64, float32, float64.

        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.

        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:

        Output (Tensor): If ``reduction`` is ``'none'``, the shape of output is same as ``input`` , else the shape of output is [1].

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.to_tensor([[0.5, 0.6, 0.7],[0.3, 0.5, 0.2]], 'float32')
            label = paddle.to_tensor([[1.0, -1.0, 1.0],[-1.0, 1.0, 1.0]], 'float32')
            output = paddle.nn.functional.soft_margin_loss(input, label)
            print(output)
            # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [0.64022040])

            input = paddle.uniform(shape=(5, 5), dtype="float32", min=0.1, max=0.8)
            label = paddle.randint(0, 2, shape=(5, 5), dtype="int64")
            label[label==0]=-1

            output = paddle.nn.functional.soft_margin_loss(input, label, reduction='none')
            print(output)
            # Tensor(shape=[5, 5], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [[1.09917796, 0.52613139, 0.56263304, 0.82736146, 0.38776723],
            #         [1.07179427, 1.11924267, 0.49877715, 1.10026348, 0.46184641],
            #         [0.84367639, 0.74795729, 0.44629076, 0.55123353, 0.77659678],
            #         [0.39465919, 0.76651484, 0.54485321, 0.76609844, 0.77166790],
            #         [0.51283568, 0.84757161, 0.78913331, 1.05268764, 0.45318675]])

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in soft_margin_loss should be 'sum', "
            "'mean' or 'none', but received %s, which is not allowed."
            % reduction
        )

    if not _non_static_mode():
        fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'soft_margin_loss'
        )
        fluid.data_feeder.check_variable_and_dtype(
            label,
            'label',
            ['int32', 'int64', 'float32', 'float64'],
            'soft_margin_loss',
        )

    if not (input.shape == label.shape):
        raise ValueError("input's shape must equal to " "label's shape")

    label = fluid.layers.cast(label, input.dtype)
    out = paddle.log(1 + paddle.exp(-label * input))

    if reduction == 'sum':
        return paddle.sum(out, name=name)
    elif reduction == 'mean':
        return paddle.mean(out, name=name)
    else:
        return out
