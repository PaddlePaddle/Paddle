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
from ...fluid.layers import bpr_loss  #DEFINE_ALIAS
from ...fluid.layers import center_loss  #DEFINE_ALIAS
#from ...fluid.layers import cross_entropy  #DEFINE_ALIAS
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
from ...fluid import core, layers
from ...fluid.framework import in_dygraph_mode, default_main_program, Variable
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.layer_helper import LayerHelper

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


def cross_entropy(input,
                  label,
                  weight=None,
                  ignore_index=-100,
                  reduction='mean'):
    """
    This operator implements the cross entropy loss function. This OP combines ``LogSoftmax``,
    and ``NLLLoss`` together.

    It is useful when training a classification problem with ``C`` classes.
    If provided, the optional argument ``weight`` should be a 1D Variable assigning
    weight to each of the classes.

    For predictions label, and target label, the loss is calculated as follows.

    .. math::

        loss_j =  -\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right), j = 1,..., K

    If weight is not ``None``:

    .. math::

        loss_j =  \\text{weight[class]}(-\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right)), j = 1,..., K

    Parameters:
        input (Variable): Input tensor, the data type is float32, float64. Shape is
	    (N, C), where C is number of classes, and if shape is more than 2D, this
	    is (N, C, D1, D2,..., Dk), k >= 1. 
        label (Variable): Label tensor, the data type is int64. Shape is (N), where each 
	    value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
	    (N, D1, D2,..., Dk), k >= 1.
        weight (Variable, optional): Weight tensor, a manual rescaling weight given
            to each class and the shape is (C). It has the same dimensions as class
	    number and the data type is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default is ``-100``.

    Returns:
        The tensor variable storing the cross_entropy_loss of input and label.

    Return type: Variable.

    Examples:
        .. code-block:: python

            import paddle
            paddle.disable_static()
            input_data = np.random.random([5, 100]).astype("float64")
            label_data = np.random.randint(0, 100, size=(5)).astype(np.int64)
            weight_data = np.random.random([100]).astype("float64")
            input =  paddle.to_tensor(input_data)
            label =  paddle.to_tensor(label_data)
            weight = paddle.to_tensor(weight_data)
            loss = paddle.nn.functional.cross_entropy(input=input, label=label, weight=weight)
            print(loss.numpy())
 
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'cross_entropy_loss')
        check_variable_and_dtype(label, 'label', ['int64'],
                                 'cross_entropy_loss')

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in cross_entropy_loss should be 'sum', 'mean' or"
            " 'none', but received %s, which is not allowed." % reduction)

    #step 1. log_softmax
    log_softmax_out = paddle.nn.functional.log_softmax(input)
    if weight is not None and not isinstance(weight, Variable):
        raise ValueError(
            "The weight' is not a Variable, please convert to Variable.")

    #step 2. nll_loss 
    input = log_softmax_out
    helper = LayerHelper('nll_loss', **locals())
    dtype = helper.input_dtype(input)

    if not in_dygraph_mode():
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'nll_loss')
        check_variable_and_dtype(label, 'label', ['int64'], 'nll_loss')

    x_shape = list(input.shape)
    n = x_shape[0]
    c = x_shape[1]
    x_dims = len(x_shape)
    if x_dims < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(
            x_dims))
    if x_dims != 2 and x_dims != 4:
        input = fluid.layers.reshape(input, shape=[n, c, 1, -1])
        label = fluid.layers.reshape(label, shape=[n, 1, -1])
        out_shape = [n] + x_shape[2:]

    inputs = {'X': input, 'Label': label}
    attrs = {'reduction': reduction, 'ignore_index': ignore_index}
    if weight is not None:
        if isinstance(weight, Variable):
            inputs['Weight'] = weight

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    total_weight = helper.create_variable_for_type_inference(dtype=input.dtype)
    outputs = {'Out': out, 'Total_weight': total_weight}

    helper.append_op(
        type='nll_loss', inputs=inputs, outputs=outputs, attrs=attrs)
    if x_dims != 2 and x_dims != 4 and reduction == 'none':
        out = layers.reshape(out, shape=out_shape)

    return out
