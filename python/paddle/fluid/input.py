#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from .framework import Variable, in_dygraph_mode
from .layer_helper import LayerHelper

__all__ = ['one_hot']


def one_hot(input, depth, allow_out_of_range=False):
    """
    This layer creates the one-hot representations for input indices.

    Args:
        input(Variable): Input indices represent locations, which takes value 1.0
            in indices, while all other locations take value 0.
        depth(scalar): An interger defining the depth of the one-hot dimension.
        allow_out_of_range(bool): A bool value indicating whether the input
            indices could be out of range [0, depth). When input indices are
            out of range, exceptions is raised if allow_out_of_range is False,
            or zero-filling representations is created if it is set True

    Returns:
        Variable: The one-hot representations of input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            one_hot_label = fluid.input.one_hot(input=label, depth=10)
    """
    helper = LayerHelper("one_hot_v2", **locals())

    one_hot_out = helper.create_variable_for_type_inference(dtype='float32')

    if in_dygraph_mode():
        inputs = {'X': input}
        attrs = {'depth': depth}
    else:
        if not isinstance(depth, Variable):
            # user attribute 
            inputs = {'X': input}
            attrs = {'depth': depth}
        else:
            depth.stop_gradient = True
            inputs = {'X': input, 'depth_tensor': depth}
            attrs = {}
    helper.append_op(
        type="one_hot_v2",
        inputs=inputs,
        attrs=attrs,
        outputs={'Out': one_hot_out},
        stop_gradient=True)
    return one_hot_out
