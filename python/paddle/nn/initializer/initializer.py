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

import paddle.fluid as fluid

__all__ = ['set_global_initializer']


def set_global_initializer(weight_init, bias_init=None):
    """
    This API is used to set up global model parameter initializer in framework.

    After this API is invoked, the global initializer will takes effect in subsequent code.

    The model parameters include ``weight`` and ``bias`` . In the framework, they correspond 
    to ``fluid.Parameter`` , which is inherited from ``fluid.Variable`` , and is a persistable Variable.
    This API only takes effect for model parameters, not for variables created through apis such as 
    :ref:`api_fluid_layers_create_global_var` , :ref:`api_fluid_layers_create_tensor`.
    
    If the initializer is also set up by ``param_attr`` or ``bias_attr`` when creating a network layer,
    the global initializer setting here will not take effect because it has a lower priority.

    If you want to cancel the global initializer in framework, please set global initializer to ``None`` .

    Args:
        weight_init (Initializer): set the global initializer for ``weight`` of model parameters.
        bias_init (Initializer, optional): set the global initializer for ``bias`` of model parameters. 
            Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            import paddle.nn as nn

            nn.initializer.set_global_initializer(nn.initializer.Uniform(), nn.initializer.Constant())
            x = paddle.static.data(name="x", shape=[1, 3, 32, 32])

            # The weight of conv1 is initialized by Uniform
            # The bias of conv1 is initialized by Constant
            conv1 = fluid.layers.conv2d(x, 5, 3)

            # If set param_attr/bias_attr too, global initializer will not take effect
            # The weight of conv2 is initialized by Xavier
            # The bias of conv2 is initialized by Normal
            conv2 = fluid.layers.conv2d(conv1, 5, 3, 
                param_attr=nn.initializer.XavierUniform(), 
                bias_attr=nn.initializer.Normal())

            # Cancel the global initializer in framework, it will takes effect in subsequent code
            nn.initializer.set_global_initializer(None)
    """

    fluid.initializer.set_global_initializer(weight_init, bias_init)
