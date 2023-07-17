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

import paddle


def shard_tensor(
    data, dtype=None, place=None, stop_gradient=True, dist_attr=None
):
    """
    Constructs a ``paddle.Tensor`` with distributed attributes from ``data``,
    which can scalar, tuple, list, numpy.ndarray, paddle.Tensor.

    If the ``data`` is already a Tensor, transform it to a Distributed Tensor.

    Args:
        data(scalar|tuple|list|ndarray|Tensor): Initial data for the tensor.
            Can be a scalar, list, tuple, numpy.ndarray, paddle.Tensor.
        dtype(str|np.dtype, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' ,
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data``
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional): The place to allocate Tensor. Can be
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs.
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.
        dist_attr(paddle.distributed.DistAttr): Specify how tensors are distributed or sliced on ProcessMesh.

    Returns:
        Tensor: A Tensor constructed from ``data`` with distributed attributes.

    Examples:

    .. code-block:: python

        import paddle
        import paddle.distributed as dist

        mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])
        dist_attr = dist.DistAttr(sharding_specs==['x', 'y'], mesh=mesh)

        # dense tensor
        a = paddle.to_tensor([[1,2,3],
                              [5,6,7]])
        # distributed tensor
        d_tensor = dist.shard_tensor(a, dist_attr=dist_attr)

        print(d_tensor)
    """
    # 1. create dense tensor
    # `paddle.to_tensor` supports both dynamic and static mode
    data = paddle.to_tensor(data)

    # 2. create dist tensor
    if paddle.in_dynamic_mode():
        return paddle.Tensor(data, dist_attr=dist_attr)
    else:
        raise NotImplementedError(
            "The `paddle.distributed.shard_tensor` for static mode will be implemented later."
        )
