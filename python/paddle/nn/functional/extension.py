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
__all__ = [
    #            'add_position_encoding',
    #            'autoincreased_step_counter',
    #            'continuous_value_model',
    #            'filter_by_instag',
    #            'linear_chain_crf',
    #            'merge_selected_rows',
    #            'multiclass_nms',
    #            'polygon_box_transform',
    #            'random_crop',
    'row_conv',
    #            'rpn_target_assign',
    #            'similarity_focus',
    #            'target_assign',
    #            'temporal_shift',
    #            'warpctc',
    #            'diag_embed'
]

from ...fluid import core, dygraph_utils
from ...fluid.framework import in_dygraph_mode
from ...fluid.layer_helper import LayerHelper
from ...fluid.layers.layer_function_generator import templatedoc


@templatedoc()
def row_conv(input, weight, act=None):
    """
    ${comment}

    Args:
        input (Variable):  the input(X) is a LodTensor or tensor, LodTensor(X) 
            supports variable  time-length input sequences. The underlying 
            tensor in this LoDTensor is a matrix with shape (T, D), where 
            T is the total time steps in this mini-batch and D is the input 
            data dimension. 
            If the input is a padded minibatch, the shape of the input is 
            (N, T, D), N is batch size, T is the max time steps in the batch,
             D is the input data dimension.
        weight (Variable): The weight. A Tensor with shape 
            (future_context_size + 1, D), where future_context_size is the 
            context size of the RowConv operator.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        ${out_comment}.

    Examples:
        .. code-block:: python

            from paddle import fluid, nn
            import paddle.fluid.dygraph as dg
            import paddle.nn.functional as F
            import numpy as np

            batch_size = 4
            time_steps = 8
            feature_size = 6
            context_size = 4
            x = np.random.randn(batch_size, time_steps, feature_size).astype(np.float32)
            weight = np.random.randn(context_size + 1, feature_size).astype(np.float32)

            place = fluid.CPUPlace()
            with dg.guard(place):
                x_var = dg.to_variable(x)
                w_var = dg.to_variable(weight)
                y_var = F.row_conv(x_var, w_var)
                y_np = y_var.numpy()

            print(y_np.shape)

            # (4, 8, 6)
    """

    if in_dygraph_mode():
        pre_act = core.ops.row_conv(input, weight)
        out = dygraph_utils._append_activation_in_dygraph(pre_act, act)
        return out
    else:
        helper = LayerHelper('row_conv', **locals())
        dtype = helper.input_dtype()

        inputs = {'X': [input], 'Filter': [weight]}
        pre_act = helper.create_variable_for_type_inference(dtype)
        outputs = {'Out': [pre_act]}
        helper.append_op(type='row_conv', inputs=inputs, outputs=outputs)
        out = helper.append_activation(pre_act)
    return out
