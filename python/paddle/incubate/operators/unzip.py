#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.layer_helper import LayerHelper


def unzip(input, lod):
    r"""

    **unzip layers**

    unzip 'input' accroding to 'lod'

    Args:
        input (Variable): The zipped input, 2-D LodTensor with shape [N, M].
        lod (Variable): The original lod of unzipped input, 1-D LodTensor with shape[K].

    Returns:
        Variable: The original unzipped tensor, 2-D LodTensor with shape[K-1, M].

    Examples:

        .. code-block:: python
          import numpy as np
          import paddle
          import paddle.fluid as fluid
          paddle.enable_static()
          input_np = np.array([
                        [1.0, 2.0, 3.0, 4.0],
                        [10.0, 20.0, 30.0, 40.0],
                        [100.0, 200.0, 300.0, 400.0]
                    ])
          lod_np = np.array([0, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12])
          input = paddle.to_tensor(input_np, "int64")
          lod = paddle.to_tensor(lod_np, "int64")

          unzipped_input = paddle.incubate.unzip(input, lod)
          '''
          unzipped_input is [
                        [1.0, 2.0, 3.0, 4.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [10.0, 20.0, 30.0, 40.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [100.0, 200.0, 300.0, 400.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]
                    ]
          '''
    """
    helper = LayerHelper('unzip', **locals())
    out = helper.create_variable(dtype=input.dtype)
    check_variable_and_dtype(
        input,
        'input',
        ['float16', 'float32', 'float64', 'int', 'bool', 'int64'],
        'unzip',
    )
    check_variable_and_dtype(lod, 'lod', ['int', 'int64'], 'unzip')
    helper.append_op(
        type='unzip', inputs={'X': [input], 'lod': [lod]}, outputs={'Y': [out]}
    )
    return out
