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

from . import core

__all__ = ['set_tf32', 'allow_tf32']


def set_tf32(on_off):
    """
    Set tf32 switch by users.

    Args:
      on_off: The param passed by usrs, indecating whether activate
      the tf32 acceleration or not.
    Returns:
      None
    Examples:
      .. code-block:: python

        import numpy as np
        import paddle
        import paddle.fluid as fluid
        import paddle.fluid.core as core

        if core.is_compiled_with_cuda():
            device = core.CUDAPlace(0)
        else:
            device = core.CPUPlace()
            fluid.tf32_switch.set_tf32(0)  # turn off
        with fluid.dygraph.guard(device):
            input_array1 = np.random.rand(4, 12, 64, 88).astype("float32")
            input_array2 = np.random.rand(4, 12, 88, 512).astype("float32") 
            data1 = fluid.dygraph.to_variable(input_array1)
            data2 = fluid.dygraph.to_variable(input_array2)
            out = paddle.matmul(data1, data2)
            expected_result = np.matmul(input_array1, input_array2)
            if np.allclose(expected_result, out.numpy(), 1e-03):
                print("Correct computation")
            else:
                print("Incorrect computation")
    """
    return core.set_switch(on_off)


def allow_tf32():
    """
    get the state of tf32 switch.

    Args:
      None
    Returns:
      True when turning it on, False when turning off
    Examples:
      .. code-block:: python
        import paddle.fluid as fluid
        import paddle.fluid.core as core
        
        if core.is_compiled_with_cuda():
            if fluid.tf32_switch.allow_tf32():
                print("tf32 acceleration is on")
            else:
                print("tf32 acceleration is off")
        else:
            pass
    """

    return core.get_switch()
