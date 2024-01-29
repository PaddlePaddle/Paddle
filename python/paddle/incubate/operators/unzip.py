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

from paddle.base.data_feeder import check_type, check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper


def unzip(input, lod, len):
    r"""

    **unzip layers**

    unzip 'input' according to 'lod'

    Args:
        input (Variable): The zipped input
        len(int): The second dim length of unzipped output.
        lod (Variable): The original lod of unzipped input, 1-D LodTensor with shape[K].

    Returns:
        Variable: The original unzipped tensor, 2-D LodTensor with shape[K-1, len].

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import numpy as np
            >>> import paddle
            >>> paddle.set_device('gpu')
            >>> input_np = np.array([1, 2, 3, 1, 2, 4])
            >>> lod_np = np.array([0, 3, 3, 3, 4, 6])
            >>> len = 4
            >>> input = paddle.to_tensor(input_np, "int64")
            >>> lod = paddle.to_tensor(lod_np, "int64")
            >>> unzipped_input = paddle.incubate.operators.unzip(input, lod, len)
            >>> print(unzipped_input)
            Tensor(shape=[5, 4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                   [[1, 2, 3, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [2, 4, 0, 0]])

    """
    helper = LayerHelper('unzip', **locals())
    out = helper.create_variable(dtype=input.dtype)
    check_variable_and_dtype(
        input,
        'input',
        [
            'float16',
            'float32',
            'float64',
            'int',
            'bool',
            'int64',
            'complex64',
            'complex128',
        ],
        'unzip',
    )
    check_variable_and_dtype(lod, 'lod', ['int', 'int64'], 'unzip')
    check_type(len, 'len', (int), 'unzip')
    attrs = {'len': len}
    helper.append_op(
        type='unzip',
        inputs={'X': [input], 'lod': [lod]},
        outputs={'Y': [out]},
        attrs=attrs,
    )
    return out
