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

import paddle

# TODO: define loss functions of neural network  
import numpy as np
import paddle
import paddle.fluid as fluid
from ...fluid.framework import core, in_dygraph_mode
from ...fluid.layers.nn import _elementwise_op_in_dygraph
from ...fluid.layers import bpr_loss  #DEFINE_ALIAS
from ...fluid.layers import center_loss  #DEFINE_ALIAS
from ...fluid.layers import cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import dice_loss  #DEFINE_ALIAS
from ...fluid.layers import iou_similarity  #DEFINE_ALIAS
from ...fluid.layers import log_loss  #DEFINE_ALIAS
from ...fluid.layers import npair_loss  #DEFINE_ALIAS
from ...fluid.layers import rank_loss  #DEFINE_ALIAS
from ...fluid.layers import reshape
from ...fluid.layers import sigmoid_cross_entropy_with_logits  #DEFINE_ALIAS
from ...fluid.layers import sigmoid_focal_loss  #DEFINE_ALIAS
from ...fluid.layers import smooth_l1  #DEFINE_ALIAS
from ...fluid.layers import softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import square_error_cost  #DEFINE_ALIAS
from ...fluid.layers import ssd_loss  #DEFINE_ALIAS
from ...fluid.layers import teacher_student_sigmoid_loss  #DEFINE_ALIAS

from ...fluid.layers import edit_distance  #DEFINE_ALIAS
from ...fluid.layers import huber_loss  #DEFINE_ALIAS
from ...fluid.layers import sampled_softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode
from ...fluid.framework import Variable

__all__ = [
    'bpr_loss',
    'center_loss',
    'cross_entropy',
    'dice_loss',
    'edit_distance',
    'huber_loss',
    'iou_similarity',
    'kl_div',
    'l1_loss',
    'log_loss',
    'mse_loss',
    'margin_ranking_loss',
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


def margin_ranking_loss(input,
                        other,
                        label,
                        margin=0.0,
                        reduction='mean',
                        name=None):
    """

    This op the calcluate the the margin rank loss between the input, other and label, use the math function as follows. 

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

    Returns: Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[1]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Examples:

        .. code-block:: python

            import numpy as np 
            import paddle 
            
            paddle.disable_static()
             
            input = paddle.to_variable(np.array([[1, 2], [3, 4]]).astype('float32'))
            other = paddle.to_variable(np.array([[2, 1], [2, 4]]).astype('float32'))
            label = paddle.to_variable(np.array([[1, -1], [-1, -1]]).astype('float32'))
            loss = paddle.nn.functional.margin_ranking_loss(input, other, label) 
            print(loss.numpy()) # [0.75]
    """
    if fluid.framework.in_dygraph_mode():
        out = core.ops.elementwise_sub(other, input)
        out = core.ops.elementwise_mul(out, label)
        if margin != 0.0:
            margin = fluid.dygraph.base.to_variable([margin], dtype=out.dtype)
            out = core.ops.elementwise_add(out, margin)
        out = core.ops.relu(out)
        if reduction == 'sum':
            return core.ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == 'mean':
            return core.ops.mean(out)
        return out

    helper = LayerHelper("margin_ranking_loss", **locals())
    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        other, 'other', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'margin_rank_loss')

    out = paddle.elementwise_sub(other, input)
    out = paddle.multiply(out, label)

    if margin != 0.0:
        margin_var = out.block.create_var(dtype=out.dtype)
        paddle.fill_constant([1], out.dtype, margin, out=margin_var)
        out = paddle.add(out, margin_var)

    result_out = helper.create_variable_for_type_inference(input.dtype)

    if reduction == 'none':
        helper.append_op(
            type="relu", inputs={"X": out}, outputs={"Out": result_out})
        return result_out
    elif reduction == 'sum':
        out = paddle.nn.functional.relu(out)
        attrs = {"dim": [0], "keep_dim": False, "reduce_all": True}
        helper.append_op(
            type="reduce_sum",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs=attrs)
        return result_out
    elif reduction == 'mean':
        out = paddle.nn.functional.relu(out)
        helper.append_op(
            type="mean",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs={})
        return result_out


def l1_loss(input, label, reduction='mean', name=None):
    """
    This operator computes the L1 Loss of Tensor ``input`` and ``label`` as follows.

    If `reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \lvert input - label\rvert

    If `reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\lvert input - label\rvert)

    If `reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\lvert input - label\rvert)

    
    Parameters:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        reduction (str, optional): Indicate the reduction to apply to the loss, 
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned; 
            If `reduction` is ``'mean'``, the reduced mean loss is returned. 
            If `reduction` is ``'sum'``, the reduced sum loss is returned. 
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the L1 Loss of Tensor ``input`` and ``label``.
            If `reduction` is ``'none'``, the shape of output loss is [N, *], the same as ``input`` .
            If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1].
    Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            
            paddle.disable_static()
            input_data = np.array([[1.5, 0.8], [0.2, 1.3]]).astype("float32")
            label_data = np.array([[1.7, 1], [0.4, 0.5]]).astype("float32")
            input = paddle.to_variable(input_data)
            label = paddle.to_variable(label_data)

            l1_loss = paddle.nn.functional.l1_loss(input, label)
            print(l1_loss.numpy())  
            # [0.35]

            l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='none')
            print(l1_loss.numpy())  
            # [[0.20000005 0.19999999]
            # [0.2        0.79999995]]

            l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='sum')
            print(l1_loss.numpy())  
            # [1.4]
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
            "received %s, which is not allowed." % reduction)

    if in_dygraph_mode():
        unreduced = _elementwise_op_in_dygraph(
            input, label, axis=-1, act='abs', op_name='elementwise_sub')
        if reduction == 'mean':
            return core.ops.mean(unreduced)
        elif reduction == 'sum':
            return core.ops.reduce_sum(unreduced, 'dim', [0], 'keep_dim', False,
                                       'reduce_all', True)
        else:
            return unreduced

    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')

    if reduction == 'sum':
        unreduced = paddle.elementwise_sub(input, label, act='abs')
        return paddle.sum(unreduced, name=name)
    elif reduction == 'mean':
        unreduced = paddle.elementwise_sub(input, label, act='abs')
        return paddle.mean(unreduced, name=name)
    else:
        return paddle.elementwise_sub(input, label, act='abs', name=name)


def nll_loss(input,
             label,
             weight=None,
             ignore_index=-100,
             reduction='mean',
             name=None):
    """
    This api returns negative log likelihood.
    See more detail in :ref:`api_nn_loss_NLLLoss` .

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
         ignore_index (int64, optional): Specifies a target value that is ignored
             and does not contribute to the input gradient.
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
                import numpy as np
                from paddle.nn.functional import nll_loss
                log_softmax = paddle.nn.LogSoftmax(axis=1)

                input_np = np.array([[0.88103855, 0.9908683 , 0.6226845 ],
                                     [0.53331435, 0.07999352, 0.8549948 ],
                                     [0.25879037, 0.39530203, 0.698465  ],
                                     [0.73427284, 0.63575995, 0.18827209],
                                     [0.05689114, 0.0862954 , 0.6325046 ]]).astype(np.float32)
                label_np = np.array([0, 2, 1, 1, 0]).astype(np.int64)

                place = paddle.CPUPlace()
                paddle.disable_static(place)
                input = paddle.to_variable(input_np)
                log_out = log_softmax(input)
                label = paddle.to_variable(label_np)
                result = nll_loss(log_out, label)
                print(result.numpy()) # [1.0720209]
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
            "'none', but received %s, which is not allowed." % reduction)

    input_shape = list(input.shape)
    input_dims = len(input_shape)
    if input_dims < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(
            input_dims))
    n = input_shape[0]
    c = input_shape[1]
    if in_dygraph_mode():
        if input_dims != 2 and input_dims != 4:
            input, _ = core.ops.reshape2(input, 'shape', [n, c, 1, -1])
            label, _ = core.ops.reshape2(label, 'shape', [n, 1, -1])
            out_shape = [n] + input_shape[2:]
        out, total_weight = core.ops.nll_loss(input, label, weight,
                                              'ignore_index', ignore_index,
                                              'reduction', reduction)
        if input_dims != 2 and input_dims != 4 and reduction == 'none':
            out, _ = core.ops.reshape2(out, 'shape', out_shape)
        return out

    helper = LayerHelper('nll_loss', **locals())

    if input_dims != 2 and input_dims != 4:
        input = reshape(input, shape=[n, c, 1, -1])
        label = reshape(label, shape=[n, 1, -1])
        out_shape = [n] + input_shape[2:]

    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'nll_loss')
    fluid.data_feeder.check_variable_and_dtype(label, 'label', ['int64'],
                                               'nll_loss')
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
    if input_dims != 2 and input_dims != 4 and reduction == 'none':
        out = reshape(out, shape=out_shape)

    return out


def kl_div(input, label, reduction='mean', name=None):
    """
    This operator calculates the Kullback-Leibler divergence loss
    between Input(X) and Input(Target). Notes that Input(X) is the
    log-probability and Input(Target) is the probability.

    KL divergence loss is calculated as follows:

    $$l(x, y) = y * (\log(y) - x)$$

    While :math:`x` is input and :math:`y` is label.

    While :attr:`reduction` is :attr:`none`, output loss is in
    the same shape as input, loss in each point is calculated 
    seperately and no reduction is applied.
    
    While :attr:`reduction` is :attr:`mean`, output loss is in
    shape of [1] and loss value is the mean value of all losses.
    
    While :attr:`reduction` is :attr:`sum`, output loss is in
    shape of [1] and loss value is the sum value of all losses.
    
    While :attr:`reduction` is :attr:`batchmean`, output loss is 
    in shape of [1] and loss value is the sum value of all losses
    divided by batch size.

    Args:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means 
             any number of additional dimensions. It's data type should be float32, float64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64.
        reduction (Tensor): Indicate how to average the loss,
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
            import numpy as np
            import paddle.nn.functional as F
            
            paddle.enable_imperative()
            
            shape = (5, 20)
            input = np.random.uniform(-10, 10, shape).astype('float32')
            target = np.random.uniform(-10, 10, shape).astype('float32')

            # 'batchmean' reduction, loss shape will be [N]
            pred_loss = F.kl_div(paddle.to_variable(input),
                                 paddle.to_variable(target), reduction='batchmean')
            # shape=[5]
            
            # 'mean' reduction, loss shape will be [1]
            pred_loss = F.kl_div(paddle.to_variable(input),
                                 paddle.to_variable(target), reduction='mean')
            # shape=[1]

            # 'sum' reduction, loss shape will be [1]
            pred_loss = F.kl_div(paddle.to_variable(input),
                                 paddle.to_variable(target), reduction='sum')
            # shape=[1]

            # 'none' reduction, loss shape is same with input shape
            pred_loss = F.kl_div(paddle.to_variable(input),
                                 paddle.to_variable(target), reduction='none')
            # shape=[5, 20]

    """
    if paddle.in_dynamic_mode():
        out = core.ops.kldiv_loss(input, label, 'reduction', reduction)
        return out

    helper = LayerHelper('kl_div', **locals())

    fluid.data_feeder.check_variable_and_dtype(input, 'input',
                                               ['float32', 'float64'], 'kl_div')
    fluid.data_feeder.check_variable_and_dtype(label, 'label',
                                               ['float32', 'float64'], 'kl_div')
    fluid.data_feeder.check_type(reduction, 'reduction', str, 'kl_div')

    loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='kldiv_loss',
        inputs={'X': input,
                'Target': label},
        outputs={'Loss': loss},
        attrs={'reduction': reduction})
    return loss


def mse_loss(input, label, reduction='mean', name=None):
    """
    This op accepts input predications and label and returns the mean square error.

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
        Tensor: The tensor tensor storing the mean square error difference of input and label.

    Return type: Tensor.
    
    Examples:

        .. code-block:: python
            import numpy as np
            import paddle


            # static graph mode
            paddle.enable_static()
            mse_loss = paddle.nn.loss.MSELoss()
            input = paddle.data(name="input", shape=[1])
            label = paddle.data(name="label", shape=[1])
            place = paddle.CPUPlace()
            input_data = np.array([1.5]).astype("float32")
            label_data = np.array([1.7]).astype("float32")

            output = mse_loss(input,label)
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            output_data = exe.run(
                paddle.static.default_main_program(),
                feed={"input":input_data, "label":label_data},
                fetch_list=[output],
                return_numpy=True)
            print(output_data)
            # [array([0.04000002], dtype=float32)]

            # dynamic graph mode
            paddle.disable_static()
            input = paddle.to_variable(input_data)
            label = paddle.to_variable(label_data)
            output = mse_loss(input, label)
            print(output.numpy())
            # [0.04000002]

    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'mse_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction))

    if not paddle.fluid.framework.in_dygraph_mode():
        paddle.fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'mse_loss')
        paddle.fluid.data_feeder.check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'mse_loss')

    if reduction == 'none':
        return paddle.fluid.layers.square(
            paddle.fluid.layers.elementwise_sub(input, label), name=name)
    elif reduction == 'mean':
        return paddle.mean(
            paddle.fluid.layers.square(
                paddle.fluid.layers.elementwise_sub(input, label)),
            name=name)
    else:
        return paddle.sum(paddle.fluid.layers.square(
            paddle.fluid.layers.elementwise_sub(input, label)),
                          name=name)
