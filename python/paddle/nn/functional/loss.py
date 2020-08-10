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
from paddle.fluid import core
from ...fluid.layers import bpr_loss  #DEFINE_ALIAS
from ...fluid.layers import center_loss  #DEFINE_ALIAS
from ...fluid.layers import cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import dice_loss  #DEFINE_ALIAS
from ...fluid.layers import iou_similarity  #DEFINE_ALIAS
from ...fluid.layers import kldiv_loss  #DEFINE_ALIAS
from ...fluid.layers import log_loss  #DEFINE_ALIAS
from ...fluid.layers import mse_loss  #DEFINE_ALIAS
from ...fluid.layers import npair_loss  #DEFINE_ALIAS
from ...fluid.layers import rank_loss  #DEFINE_ALIAS
from ...fluid.layers import sigmoid_cross_entropy_with_logits  #DEFINE_ALIAS
from ...fluid.layers import sigmoid_focal_loss  #DEFINE_ALIAS
from ...fluid.layers import smooth_l1  #DEFINE_ALIAS
from ...fluid.layers import softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import square_error_cost  #DEFINE_ALIAS
from ...fluid.layers import ssd_loss  #DEFINE_ALIAS
from ...fluid.layers import teacher_student_sigmoid_loss  #DEFINE_ALIAS

from ...fluid.layers import edit_distance  #DEFINE_ALIAS
from ...fluid.layers import huber_loss  #DEFINE_ALIAS
from ...fluid.layers import margin_rank_loss  #DEFINE_ALIAS
from ...fluid.layers import sampled_softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode

__all__ = [
    'bpr_loss',
    'center_loss',
    'cross_entropy',
    'dice_loss',
    'edit_distance',
    'huber_loss',
    'iou_similarity',
    'kldiv_loss',
    'log_loss',
    'margin_rank_loss',
    'mse_loss',
    #       'nce',
    'nll_loss',
    'npair_loss',
    'rank_loss',
    'sampled_softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'sigmoid_focal_loss',
    'smooth_l1',
    'softmax_with_cross_entropy',
    'square_error_cost',
    'ssd_loss',
    'teacher_student_sigmoid_loss'
]


def nll_loss(x,
             label,
             weight=None,
             ignore_index=-100,
             reduction='mean',
             name=None):
    """
    This api returns negative log likelihood.
    See more detail in :ref:`api_nn_loss_NLLLoss` .

    Parameters:
         x (Tensor): Input tensor, the data type is float32, float64.
         label (Tensor): Label tensor, the data type is int64.
         weight (Tensor, optional): Weight tensor, a manual rescaling weight given
             to each class. If given, it has to be a Tensor of size `C`. Otherwise,
             it treated as if having all ones. the data type is
             float32, float64, Default is ``'None'``.
         ignore_index (int64, optional): Specifies a target value that is ignored
             and does not contribute to the input gradient.
         reduction (str, optional): Indicate how to average the loss,
             the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
             If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                :attr:`reduction` is ``'sum'``, the reduced sum loss is returned;
                :attr:`reduction` is ``'none'``, no reduction will be apllied.
             Default is ``'mean'``.
         name (str, optional): Name for the operation (optional, default is None).
             For more information, please refer to :ref:`api_guide_Name`.

    Returns:
         The tensor variable storing the nll_loss.

    Examples:
        .. code-block:: python

                import paddle
                import numpy as np
                from paddle.nn.functional import nll_loss
                log_softmax = paddle.nn.LogSoftmax(axis=1)

                x_np = np.random.random(size=(10, 10)).astype(np.float32)
                label_np = np.random.randint(0, 10, size=(10,)).astype(np.int64)

                place = paddle.CPUPlace()

                # imperative mode
                paddle.enable_imperative(place)
                x = paddle.imperative.to_variable(x_np)
                log_out = log_softmax(x)
                label = paddle.imperative.to_variable(label_np)
                imperative_result = nll_loss(log_out, label)
                print(imperative_result.numpy())
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
            "'none', but received %s, which is not allowed." % reduction)

    if in_dygraph_mode():
        x_shape = list(core.ops.shape(x))
        x_dims = len(x_shape)
        if x_dims < 2:
            raise ValueError('Expected 2 or more dimensions (got {})'.format(
                x_dims))
        n = x_shape[0]
        c = x_shape[1]
        if x_dims != 2 and x_dims != 4:
            x, _ = core.ops.reshape2(x, 'shape', [n, c, 1, -1])
            label, _ = core.ops.reshape2(label, 'shape', [n, 1, -1])
            out_shape = [n] + x_shape[2:]
        out, total_weight = core.ops.nll_loss(x, label, weight, 'ignore_index',
                                              ignore_index, 'reduction',
                                              reduction)
        if x_dims != 2 and x_dims != 4 and reduction == 'none':
            out, _ = core.ops.reshape2(out, 'shape', out_shape)
        return out

    helper = LayerHelper('nll_loss', **locals())
    x_shape = list(x.shape)
    x_dims = len(x_shape)
    if x_dims < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(
            x_dims))
    n = x_shape[0]
    c = x_shape[1]

    if x_dims != 2 and x_dims != 4:
        x = paddle.reshape(x, shape=[n, c, 1, -1])
        label = paddle.reshape(label, shape=[n, 1, -1])
        out_shape = [n] + x_shape[2:]

    fluid.data_feeder.check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                               'nll_loss')
    fluid.data_feeder.check_variable_and_dtype(label, 'label', ['int64'],
                                               'nll_loss')
    inputs = {'X': x, 'Label': label}
    attrs = {'reduction': reduction, 'ignore_index': ignore_index}
    if weight is not None:
        if isinstance(weight, paddle.Variable):
            inputs['Weight'] = weight

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    total_weight = helper.create_variable_for_type_inference(dtype=x.dtype)
    outputs = {'Out': out, 'Total_weight': total_weight}

    helper.append_op(
        type='nll_loss', inputs=inputs, outputs=outputs, attrs=attrs)
    if x_dims != 2 and x_dims != 4 and reduction == 'none':
        out = paddle.reshape(out, shape=out_shape)

    return out
