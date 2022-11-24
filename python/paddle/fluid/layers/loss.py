# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from functools import partial, reduce
import paddle
from paddle.utils import deprecated
from . import nn
from .layer_function_generator import templatedoc
from ..layer_helper import LayerHelper
from ..framework import (
    Variable,
    _non_static_mode,
    static_only,
    _in_legacy_dygraph,
    in_dygraph_mode,
)
from .. import core
from ..data_feeder import check_variable_and_dtype, check_type
from ..param_attr import ParamAttr
from ..initializer import NumpyArrayInitializer, Constant
from .. import core
import warnings
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'cross_entropy',
    'square_error_cost',
    'warpctc',
    'nce',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
]

kIgnoreIndex = -100


def cross_entropy(input, label, soft_label=False, ignore_index=kIgnoreIndex):
    r"""
    :alias_main: paddle.nn.functional.cross_entropy
        :alias: paddle.nn.functional.cross_entropy,paddle.nn.functional.loss.cross_entropy
        :old_api: paddle.fluid.layers.cross_entropy

    This operator computes the cross entropy between input and label. It
    supports both hard-label and and soft-label cross entropy computation.

    1. Hard-label cross entropy: if soft_label=False, :math:`label[i_1, i_2, ..., i_k]`
       is the hard label of each sample.

        .. math::

           output[i_1, i_2, ..., i_k]=-log(input[i_1, i_2, ..., i_k, j]), label[i_1, i_2, ..., i_k] = j, j != ignore\_index

    2. Soft-label cross entropy: if soft_label=True,  :math:`label[i_1, i_2, ..., i_k, j]`
       is the soft label of each sample corresponding to the j-th class.

        .. math::

           output[i_1, i_2, ..., i_k]= -\sum_{j}label[i_1,i_2,...,i_k,j]*log(input[i_1, i_2, ..., i_k,j])

    Args:
        input (Variable): a multidimensional Tensor with shape
                :math:`[N_1, N_2, ..., N_k, D]`, where the last dimension D is
                the class number. The data type should be float32 or float64.
        label (Variable): label value corresponding to input. If
                soft_label=False, the dimension of label should be :math:`[N_1, N_2, ..., N_k]`
                or :math:`[N_1, N_2, ..., N_k, 1]` , and its data type should be int64,
                and the value must be inside [0, D). If soft_label=True, the shape,
                data type of label should be the same with input, and the sum of
                soft label value of each sample should be 1.
        soft_label (bool): indicate whether label is soft. Default False, meaning that
                the label is hard. If soft_label=True, the label is soft.
        ignore_index (int): specify an ignorable label value. The ignored label would be
                omitted when computing. If it is a negative integer, no label would
                be ignored. Only valid when soft_label=False. Default -100.

    Returns:
         A Variable holding Tensor representing the cross entropy, whose data type is the same with input.
         If soft_label=False, the shape of output is the same with label.
         If soft_label=True, the shape of output is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            class_num = 7
            x = fluid.data(name='x', shape=[None, 3, 10], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            predict = fluid.layers.fc(input=x, size=class_num, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
    """
    if not soft_label:
        return cross_entropy2(input, label, ignore_index)

    if _non_static_mode():
        return _legacy_C_ops.cross_entropy(
            input, label, "soft_label", soft_label, "ignore_index", ignore_index
        )

    inputs = {'X': [input], 'Label': [label]}
    attrs = {"soft_label": soft_label, "ignore_index": ignore_index}

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'cross_entropy'
    )
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy', inputs=inputs, outputs={'Y': [out]}, attrs=attrs
    )
    return out


def cross_entropy2(input, label, ignore_index=kIgnoreIndex):
    if _non_static_mode():
        loss, _, _ = _legacy_C_ops.cross_entropy2(
            input, label, 'ignore_index', ignore_index
        )
        return loss

    inputs = {'X': [input], 'Label': [label]}
    attrs = {'ignore_index': ignore_index}
    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'cross_entropy2'
    )
    helper = LayerHelper('cross_entropy2', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    xshape = helper.create_variable_for_type_inference(dtype=input.dtype)
    match_x = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy2',
        inputs=inputs,
        outputs={'Y': [out], 'MatchX': [match_x], 'XShape': [xshape]},
        attrs=attrs,
    )
    return out


def square_error_cost(input, label):
    r"""

    Accept input predictions and target label and returns the
    squared error cost.

    For predictions label, and target label, the equation is:

    .. math::

        Out = (input - label)^2

    Parameters:
        input (Tensor): Input tensor, the data type should be float32.
        label (Tensor): Label tensor, the data type should be float32.

    Returns:
        Tensor, The tensor storing the element-wise squared
        error difference between input and label.

    Examples:

        .. code-block:: python

            import paddle
            input = paddle.to_tensor([1.1, 1.9])
            label = paddle.to_tensor([1.0, 2.0])
            output = paddle.nn.functional.square_error_cost(input, label)
            print(output)
            # [0.01, 0.01]

    """
    return paddle.nn.functional.square_error_cost(input, label)


def warpctc(
    input,
    label,
    blank=0,
    norm_by_times=False,
    input_length=None,
    label_length=None,
):
    """
    An operator integrating the open source Warp-CTC library
    (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation is
    interated to the Warp-CTC library to normalize values for each row of the
    input tensor.

    Args:
       input (Variable): The unscaled probabilities of variable-length sequences,
         which is a 2-D Tensor with LoD information, or a 3-D Tensor without Lod
         information. When it is a 2-D LodTensor, its shape is
         `[Lp, num_classes + 1]`, where `Lp` is the sum of all input
         sequences' length and `num_classes` is the true number of classes.
         (not including the blank label). When it is a 3-D Tensor, its shape
         is `[max_logit_length, batch_size, num_classes + 1]`,
         where `max_logit_length` is the longest length of
         input logit sequence. The data type should be float32 or float64.
       label (Variable): The ground truth of variable-length sequence,
         which must be a 2-D Tensor with LoD information or a 3-D Tensor without
         LoD information, needs to be consistent with the coressponding input.
         When it is a 2-D LoDTensor, its shape is `[Lg, 1]`, where `Lg` is the sum
         of all labels' length. When it is a 3-D Tensor, its shape is
         `[batch_size, max_label_length]`, where `max_label_length` is the longest
         length of label sequence. Data type must be int32.
       blank (int, default 0): The blank label index of Connectionist
         Temporal Classification (CTC) loss, which is in the
         half-opened interval `[0, num_classes + 1)`. The data type must be int32.
       norm_by_times(bool, default false): Whether to normalize the gradients
         by the number of time-step, which is also the sequence's length.
         There is no need to normalize the gradients if warpctc layer was
         followed by a mean_op.
       input_length(Variable): The length for each input sequence if it is
         of Tensor type, it should have shape `[batch_size]` and dtype int64.
       label_length(Variable): The length for each label sequence if it is
         of Tensor type, it should have shape `[batch_size]` and dtype int64.

    Returns:
        Variable: The Connectionist Temporal Classification (CTC) loss,
        which is a 2-D Tensor with the shape `[batch_size, 1]`.
        The date type is the same as input.

    Examples:

        .. code-block:: python

            # using LoDTensor
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            # lengths of logit sequences
            seq_lens = [2,6]
            # lengths of label sequences
            label_lens = [2,3]
            # class num
            class_num = 5

            paddle.enable_static()
            logits = fluid.data(name='logits',shape=[None, class_num+1],
                                 dtype='float32',lod_level=1)
            label = fluid.data(name='label', shape=[None, 1],
                               dtype='int32', lod_level=1)
            cost = fluid.layers.warpctc(input=logits, label=label)
            place = fluid.CPUPlace()
            x = fluid.create_lod_tensor(
                     np.random.rand(np.sum(seq_lens), class_num+1).astype("float32"),
                     [seq_lens], place)
            y = fluid.create_lod_tensor(
                     np.random.randint(0, class_num, [np.sum(label_lens), 1]).astype("int32"),
                     [label_lens], place)
            exe = fluid.Executor(place)
            output= exe.run(fluid.default_main_program(),
                            feed={"logits": x,"label": y},
                            fetch_list=[cost.name])
            print(output)

        .. code-block:: python

            # using Tensor
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            # length of the longest logit sequence
            max_seq_length = 5
            #length of the longest label sequence
            max_label_length = 3
            # number of logit sequences
            batch_size = 16
            # class num
            class_num = 5
            paddle.enable_static()
            logits = fluid.data(name='logits',
                           shape=[max_seq_length, batch_size, class_num+1],
                           dtype='float32')
            logits_length = fluid.data(name='logits_length', shape=[None],
                             dtype='int64')
            label = fluid.data(name='label', shape=[batch_size, max_label_length],
                           dtype='int32')
            label_length = fluid.data(name='labels_length', shape=[None],
                             dtype='int64')
            cost = fluid.layers.warpctc(input=logits, label=label,
                            input_length=logits_length,
                            label_length=label_length)
            place = fluid.CPUPlace()
            x = np.random.rand(max_seq_length, batch_size, class_num+1).astype("float32")
            y = np.random.randint(0, class_num, [batch_size, max_label_length]).astype("int32")
            exe = fluid.Executor(place)
            output= exe.run(fluid.default_main_program(),
                            feed={"logits": x,
                                  "label": y,
                                  "logits_length": np.array([max_seq_length]*batch_size).astype("int64"),
                                  "labels_length": np.array([max_label_length]*batch_size).astype("int64")},
                                  fetch_list=[cost.name])
            print(output)
    """
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
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], "warpctc")
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


# FIXME(wuyi): let docstring_checker.py understand @autodoc.
# For now, the comments in c++ use types like Tensor, but in python side
# the type is often "Variable", and arguments may vary.
@static_only
@templatedoc(op_type="nce")
def nce(
    input,
    label,
    num_total_classes,
    sample_weight=None,
    param_attr=None,
    bias_attr=None,
    num_neg_samples=None,
    name=None,
    sampler="uniform",
    custom_dist=None,
    seed=0,
    is_sparse=False,
):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input (Tensor): Input tensor, 2-D tensor with shape [batch_size, dim],
            and data type is float32 or float64.
        label (Tensor): Input label, 2-D tensor with shape [batch_size, num_true_class],
            and data type is int64.
        num_total_classes (int):${num_total_classes_comment}.
        sample_weight (Tensor|None): A Tensor of shape [batch_size, 1]
            storing a weight for each sample. The default weight for each
            sample is 1.0.
        param_attr (ParamAttr|None): To specify the weight parameter attribute.
            Default: None, which means the default weight parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr|None): To specify the bias parameter attribute.
            Default: None, which means the default bias parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        num_neg_samples (int): ${num_neg_samples_comment}.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        sampler (str, optional): The sampler used to sample class from negative classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (nd.array|None): A numpy ndarray with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probability of i-th class to be sampled.
                       default: None.
        seed (int, optional): The seed used in sampler. Default 0, means no random seed.
        is_sparse(bool, optional): The flag indicating whether to use sparse update,
            the weight@GRAD and bias@GRAD will be changed to SelectedRows. Default False.

    Returns:
        Tensor: The output nce loss.

    Examples:
        .. code-block:: python


            import paddle
            import numpy as np

            paddle.enable_static()

            window_size = 5
            words = []
            for i in range(window_size):
                words.append(paddle.static.data(
                    name='word_{0}'.format(i), shape=[-1, 1], dtype='int64'))

            dict_size = 10000
            label_word = int(window_size / 2) + 1

            embs = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb = paddle.static.nn.embedding(input=words[i], size=[dict_size, 32],
                                    param_attr='embed', is_sparse=True)
                embs.append(emb)

            embs = paddle.concat(x=embs, axis=1)
            loss = paddle.static.nn.nce(input=embs, label=words[label_word],
                        num_total_classes=dict_size, param_attr='nce.w_0',
                        bias_attr='nce.b_0')

            #or use custom distribution
            dist = np.array([0.05,0.5,0.1,0.3,0.05])
            loss = paddle.static.nn.nce(input=embs, label=words[label_word],
                    num_total_classes=5, param_attr='nce.w_1',
                    bias_attr='nce.b_1',
                    num_neg_samples=3,
                    sampler="custom_dist",
                    custom_dist=dist)
    """
    helper = LayerHelper('nce', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'nce')
    check_variable_and_dtype(label, 'label', ['int64'], 'nce')

    dim = input.shape[1]
    num_true_class = label.shape[1]
    w = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_total_classes, dim],
        is_bias=False,
        dtype=input.dtype,
    )
    inputs = {}
    if helper.bias_attr:
        b = helper.create_parameter(
            attr=helper.bias_attr,
            shape=[num_total_classes, 1],
            is_bias=True,
            dtype=input.dtype,
        )
        inputs['Bias'] = b
    cost = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_logits = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_labels = helper.create_variable_for_type_inference(dtype=label.dtype)

    inputs['Input'] = input
    inputs['Label'] = label
    inputs['Weight'] = w
    inputs['SampleWeight'] = sample_weight if sample_weight is not None else []

    if sampler == "uniform":
        sampler = 0
    elif sampler == "log_uniform":
        sampler = 1
    elif sampler == "custom_dist":
        assert custom_dist is not None

        custom_dist_len = num_total_classes
        alias_probs_ = [0] * custom_dist_len
        alias_ = [0] * custom_dist_len
        bigs = []
        littles = []
        for i in range(custom_dist_len):
            normal_prob = custom_dist[i] * custom_dist_len
            if normal_prob - 1.0 > 0:
                bigs.append((i, normal_prob))
            elif 1.0 - normal_prob > 0:
                littles.append((i, normal_prob))
            else:
                alias_probs_[i] = normal_prob
                alias_[i] = -1

        while len(bigs) and len(littles):
            big = bigs.pop(0)
            little = littles.pop(0)

            big_idx = big[0]
            big_prob = big[1]

            alias_probs_[little[0]] = little[1]
            alias_[little[0]] = big_idx
            big_left = big[1] + little[1] - 1
            if big_left - 1.0 > 0:
                bigs.append((big_idx, big_left))
            elif 1.0 - big_left > 0:
                littles.append((big_idx, big_left))
            else:
                alias_probs_[big_idx] = big_left
                alias_[big_idx] = -1

        if len(bigs):
            big = bigs.pop(0)
            alias_probs_[big[0]] = 1.0
            alias_[big[0]] = -1
        if len(littles):
            little = littles.pop(0)
            alias_probs_[little[0]] = 1.0
            alias_[little[0]] = -1

        def _init_by_numpy_array(numpy_array):
            ret = helper.create_parameter(
                attr=ParamAttr(),
                shape=numpy_array.shape,
                dtype=numpy_array.dtype,
                default_initializer=NumpyArrayInitializer(numpy_array),
            )
            ret.stop_gradient = True
            return ret

        inputs['CustomDistProbs'] = _init_by_numpy_array(
            np.array(custom_dist).astype('float32')
        )
        inputs['CustomDistAlias'] = _init_by_numpy_array(
            np.array(alias_).astype('int32')
        )
        inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
            np.array(alias_probs_).astype('float32')
        )
        sampler = 2
    else:
        raise Exception("Unsupported sampler type.")

    if num_neg_samples is None:
        num_neg_samples = 10
    else:
        num_neg_samples = int(num_neg_samples)

    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )

    attrs = {
        'num_total_classes': int(num_total_classes),
        'num_neg_samples': num_neg_samples,
        'seed': seed,
        'sampler': sampler,
        'is_sparse': is_sparse,
        'remote_prefetch': remote_prefetch,
    }

    helper.append_op(
        type='nce',
        inputs=inputs,
        outputs={
            'Cost': cost,
            'SampleLogits': sample_logits,
            'SampleLabels': sample_labels,
        },
        attrs=attrs,
    )
    return cost / (num_neg_samples + 1)


def softmax_with_cross_entropy(
    logits,
    label,
    soft_label=False,
    ignore_index=kIgnoreIndex,
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

        loss_j =  -\\text{logits}_{label_j} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{logits}_i)\\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::

        loss_j =  -\\sum_{i=0}^{K}\\text{label}_i
        \\left(\\text{logits}_i - \\log\\left(\\sum_{i=0}^{K}
        \\exp(\\text{logits}_i)\\right)\\right), j = 1,...,K

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::

        max_j &= \\max_{i=0}^{K}{\\text{logits}_i}

        log\\_max\\_sum_j &= \\log\\sum_{i=0}^{K}\\exp(logits_i - max_j)

        softmax_j &= \\exp(logits_j - max_j - {log\\_max\\_sum}_j)

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
            import numpy as np

            data = np.random.rand(128).astype("float32")
            label = np.random.rand(1).astype("int64")
            data = paddle.to_tensor(data)
            label = paddle.to_tensor(label)
            linear = paddle.nn.Linear(128, 100)
            x = linear(data)
            out = paddle.nn.functional.softmax_with_cross_entropy(logits=x, label=label)
            print(out)
    """
    return paddle.nn.functional.loss.fluid_softmax_with_cross_entropy(
        logits,
        label,
        soft_label,
        ignore_index,
        numeric_stable_mode,
        return_softmax,
        axis,
    )


def identity_loss(x, reduction="none"):
    r"""Marks a tensor as being part of the loss calculation for IPU.

    This operator is used to handle on the (final) loss of a model so that
    it is used as the start of backpropagation.

    When `reduction` is `none`, return raw `Out`.

    When `reduction` is `mean`, return

    .. math::
        Out = MEAN(Out)

    When `reduction` is `sum`, return

    .. math::
        Out = SUM(Out)

    Parameters:
        x (Variable): The input tensor. The shapes is [N, *], where N is batch size and `*` means any number of
             additional dimensions. It's data type should be float32, float64 on CPU and float16, float32 on IPU.
        reduction(str|int, optional): Reduce the loss output. Supported string values are: 'sum', 'mean', 'none'
                            the corresponding int values are 0, 1, 2 respectively. The default value is "none".

    Returns:
        Variable: The loss ``Tensor`` with the specified reduction applied.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            loss = fluid.data(name="loss", shape=[-1, 1], dtype="float32")
            out = paddle.incubate.identity_loss(loss, reduction=1)
    """
    if isinstance(reduction, str):
        reduction = {"sum": 0, "mean": 1, "none": 2}.get(reduction.lower())
        if reduction is None:
            raise Exception("Unsupported reduction type.")

    if _non_static_mode():
        return _legacy_C_ops.identity_loss(x, "reduction", reduction)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "identity_loss")
    attrs = {'reduction': reduction}
    helper = LayerHelper('identity_loss', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="identity_loss", inputs={"X": x}, outputs={"Out": out}, attrs=attrs
    )
    return out


@templatedoc()
def sigmoid_cross_entropy_with_logits(
    x, label, ignore_index=kIgnoreIndex, name=None, normalize=False
):
    """

    ${comment}

    Args:
        x(Tensor): a 2-D tensor with shape N x D, where N is the batch size and
                D is the number of classes. This input is a tensor of logits computed
                by the previous operator. Logits are unscaled log probabilities given
                as log(p/(1-p)) The data type should be float32 or float64.
        label (Tensor): a 2-D tensor of the same type and shape as X.
                This input is a tensor of probabalistic labels for each logit.
        ignore_index(int): Specifies a target value that is ignored and
                does not contribute to the input gradient.
        name(str|None): The default value is None.  Normally there is
            no need for user to set this property.  For more information,
            please refer to :ref:`api_guide_Name`
        normalize(bool): If true, divide the output by the number of
            targets != ignore_index.

    Returns:
        out(Tensor): ${out_comment}

    Examples:
        .. code-block:: python


            import paddle

            input = paddle.rand(shape=[10], dtype='float32')
            label = paddle.rand(shape=[10], dtype='float32')
            loss = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(input, label,
                                                            ignore_index=-1, normalize=True)
            print(loss)
    """

    if in_dygraph_mode():
        return _C_ops.sigmoid_cross_entropy_with_logits(
            x, label, normalize, int(ignore_index)
        )
    check_variable_and_dtype(
        x,
        'input',
        ['float16', 'float32', 'float64'],
        'sigmoid_cross_entropy_with_logits',
    )

    helper = LayerHelper("sigmoid_cross_entropy_with_logits", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="sigmoid_cross_entropy_with_logits",
        inputs={"X": x, "Label": label},
        attrs={"ignore_index": ignore_index, 'normalize': normalize},
        outputs={"Out": out},
    )
    return out
