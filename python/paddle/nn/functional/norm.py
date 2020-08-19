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

# TODO: define normalization api  
import paddle
import paddle.fluid as fluid
from ...fluid.data_feeder import check_variable_and_dtype, check_type
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, core
from ...fluid.layers import l2_normalize  #DEFINE_ALIAS
from ...fluid.layers import lrn  #DEFINE_ALIAS

__all__ = [
    #       'batch_norm',
    #       'data_norm',
    #       'group_norm',
    #       'instance_norm',
    'l2_normalize',
    #       'layer_norm',
    'lrn',
    'normalize',
    #       'spectral_norm'
]


def normalize(x, p=2, axis=1, epsilon=1e-12, name=None):
    """
    This op normalizes ``x`` along dimension ``axis`` using :math:`L_p` norm. This layer computes

    .. math::

        y = \frac{x}{ \max\left( \lvert \lvert x \rvert \rvert_p, epsilon\right) }
    
    .. math::
        \lvert \lvert x \rvert \rvert_p = \left(\sum_i {\lvert x_i\rvert^p}  \right)^{1/p}

    where, :math:`\sum_i{\lvert x_i\rvert^p}` is calculated along the ``axis`` dimension.


    Args:
        x (Tensor): The input tensor could be N-D tensor, and the input data type could be float32 or float64.
        p (float|int, optional): The exponent value in the norm formulation. Default: 2
        axis (int, optional): The axis on which to apply normalization. If ``x`` is 1-D tensor, ``axis`` is fixed to 0. If `axis < 0`, \
            the dimension to normalization is `x.ndim + axis`. -1 is the last dimension.
        epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Default is 1e-12.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output has the same shape and data type with ``x``.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F

            paddle.disable_static()
            x = np.arange(6, dtype=np.float32).reshape(2,3)
            x = paddle.to_variable(x)
            y = F.normalize(x)
            print(y.numpy())
            # [[0.         0.4472136  0.8944272 ]
            # [0.42426404 0.5656854  0.7071067 ]]

            y = F.normalize(x, p=1.5)
            print(y.numpy())
            # [[0.         0.40862012 0.81724024]
            # [0.35684016 0.4757869  0.5947336 ]]

            y = F.normalize(x, axis=0)
            print(y.numpy())
            # [[0.         0.24253564 0.37139067]
            # [1.         0.97014254 0.9284767 ]]
    """
    if len(x.shape) == 1:
        axis = 0
    if in_dygraph_mode():
        eps = fluid.dygraph.base.to_variable([epsilon], dtype=x.dtype)
        out = core.ops.p_norm(x, 'axis', axis, 'porder',
                              float(p), 'keepdim', True, 'epsilon', epsilon)
        return x / core.ops.elementwise_max(out, eps)

    check_type(p, 'p', (float, int), 'normalize')
    check_type(axis, 'axis', (int), 'normalize')
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'normalize')

    attrs = {
        'axis': axis,
        'porder': float(p),
        'keepdim': True,
        'epsilon': epsilon,
    }
    helper = LayerHelper('p_norm', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='p_norm', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    eps = out.block.create_var(dtype=out.dtype)
    paddle.fill_constant([1], out.dtype, epsilon, out=eps)
    return paddle.elementwise_div(x, paddle.maximum(out, eps), name=name)
