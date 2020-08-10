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
import paddle.fluid as fluid
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
from ...fluid.layers import smooth_l1_v2  #DEFINE_ALIAS
from ...fluid.layers import softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import square_error_cost  #DEFINE_ALIAS
from ...fluid.layers import ssd_loss  #DEFINE_ALIAS
from ...fluid.layers import teacher_student_sigmoid_loss  #DEFINE_ALIAS

from ...fluid.layers import edit_distance  #DEFINE_ALIAS
from ...fluid.layers import huber_loss  #DEFINE_ALIAS
from ...fluid.layers import margin_rank_loss  #DEFINE_ALIAS
from ...fluid.layers import sampled_softmax_with_cross_entropy  #DEFINE_ALIAS

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
    'smooth_l1_loss',
    'softmax_with_cross_entropy',
    'square_error_cost',
    'ssd_loss',
    'teacher_student_sigmoid_loss'
]


def smooth_l1_loss(x, label, reduction='mean'):
    """
	:alias_main: paddle.nn.functional.smooth_l1_loss
	:alias: paddle.nn.functional.smooth_l1_loss,paddle.nn.functional.loss.smooth_l1_loss

    This operator is calculate smooth_l1_loss. Creates a criterion that uses a squared term if the absolute element-wise error falls below 1 
    and an L1 term otherwise. In some cases it can prevent exploding gradients. Also known as the Huber loss:

    .. math::

         loss(x,y)=\\frac{1}{n}\\sum_{i}z_i


    where z_i is given by:

    .. math::

         \\mathop{z_i}=\\left\\{\\begin{array}{rcl}
        0.5(x_i - y_i) & & {if |x_i - y_i| > 1} \\\\
        |x_i - y_i| - 0.5 & & {otherwise}
        \\end{array} \\right.

    Parameters:
        x (Variable): Input tensor, the data type is float32. Shape is
        (N, C), where C is number of classes, and if shape is more than 2D, this
        is (N, C, D1, D2,..., Dk), k >= 1.
        label (Variable): Label tensor, the data type is float32. The shape of label
        is the same as the shape of x.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.

    Returns:
        The tensor variable storing the smooth_l1_loss of x and label.

    Return type: Variable.

    Examples:
        .. code-block:: python

            # declarative mode
            import paddle
            import paddle.fluid as fluid
            import numpy as np
            x = fluid.layers.data(name="x", shape=[-1, 3], dtype="float32")
            label = fluid.layers.data(name="label", shape=[-1, 3], dtype="float32")
            result = paddle.nn.functioanl.smooth_l1_loss(x,label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            x = np.random.rand(3,3).astype("float32")
            label = np.random.rand(3,3).astype("float32")
            output= exe.run(feed={"x": x, "label": label},
                            fetch_list=[result])
            print(output)

            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                x = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                weight = dg.to_variable(weight_data)
                output = paddle.nn.functioanl.smooth_l1_loss(x,label)
                print(output.numpy())
    """
    fluid.data_feeder.check_variable_and_dtype(x, 'x', ['float32'],
                                               'smooth_l1_loss')
    fluid.data_feeder.check_variable_and_dtype(label, 'label', ['float32'],
                                               'smooth_l1_loss')

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in smooth_l1_loss should be 'sum', 'mean' or"
            " 'none', but received %s, which is not allowed." % reduction)
    out = smooth_l1_v2(x, label)
    if reduction == 'none':
        return out
    reduce_op = 'reduce_mean'
    if reduction == 'sum':
        reduce_op = 'reduce_sum'
    return getattr(fluid.layers, reduce_op)(out)
