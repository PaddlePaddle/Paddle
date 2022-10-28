#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
from paddle.fluid.framework import (
    _in_legacy_dygraph,
    in_dygraph_mode,
    dygraph_only,
)

__all__ = []


@dygraph_only
def to_uva_tensor(x, gpu_id=0):
    """
    Graph Storage Optimization API.

    This API adopted UVA(Unified Virtual Addressing) technology. UVA provides a single
    virtual memory address space for all memory in the system, which means we can access
    CPU memory from GPU. In this way, we can virtually expand the GPU memory, but with
    faster speed compared with the IO of copying data from CPU to GPU. This API is useful
    especially in large graph training domain.

    Note:
        This API is dygraph-only, and should be used under gpu version.

    Args:
        x (numpy.ndarray): The numpy ndarray data, which will be used to generate uva tensor.
        gpu_id (int): `gpu_id` means which gpu we put the UVA tensor on. Default value is 0.

    Returns:
        out (Tensor): The output UVA tensor. And the shape and dtype should be the same with `x`.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            if paddle.device.is_compiled_with_cuda():
                x = np.ones([4, 5])
                y = paddle.geometric.to_uva_tensor(x)

            # Although we see that y' place is Place(gpu:0), it is actually store on CPU.
            # Tensor(shape=[4, 5], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            #        [[1., 1., 1., 1., 1.],
            #        [1., 1., 1., 1., 1.],
            #        [1., 1., 1., 1., 1.],
            #        [1., 1., 1., 1., 1.]])

    """

    if not isinstance(x, np.ndarray):
        raise ValueError("The input x should be numpy.ndarray")

    if in_dygraph_mode():
        return paddle.fluid.core.eager.to_uva_tensor(x, gpu_id)

    if _in_legacy_dygraph():
        return paddle.fluid.core.to_uva_tensor(x, gpu_id)
