# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import copy
=======
import paddle
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

# TODO: check the hooks of tensor
# TODO: check serializing named tensor
# TODO: check influence on autograd
<<<<<<< HEAD
import sys
import threading
from collections import OrderedDict
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork

import paddle
=======
import os
import sys
import warnings
import math
import copy
import threading
import multiprocessing
from multiprocessing.util import register_after_fork
from multiprocessing.reduction import ForkingPickler

from collections import OrderedDict
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def _supported_check():
    if sys.platform != "linux":
        # warnings.warn("`paddle.multiprocessing` only support linux for now, "
        #               " import this will not take any effect !")

        return False

<<<<<<< HEAD
    return True


class _LRUSharedCache(OrderedDict):
    def __init__(self):
        self.limit = 128
        self._after_fork()
        register_after_fork(self, _LRUSharedCache._after_fork)
=======
    if not sys.version_info >= (3, 4):
        warnings.warn("Use `paddle.multiprocessing` to share paddle tensor "
                      "requires python version greater than 3.4 ."
                      " `paddle.multiprocessing` will not take any effect !!!")
        return False

    return True


class LRUSharedCache(OrderedDict):

    def __init__(self):
        self.limit = 128
        self._after_fork()
        register_after_fork(self, LRUSharedCache._after_fork)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _after_fork(self):
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            try:
                value = super().pop(key)
                super().__setitem__(key, value)
                return value
            except KeyError:
                return None

    def __setitem__(self, key, value):
        with self.lock:
            try:
                super().__delitem__(key)
            except KeyError:
                if len(self) >= self.limit:
                    super().popitem(last=False)
            super().__setitem__(key, value)


<<<<<<< HEAD
shared_cache = _LRUSharedCache()


def _cuda_from_cache(key):
=======
shared_cache = LRUSharedCache()


def cuda_from_cache(key):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    lodtensor = shared_cache.get(key)
    if lodtensor is None:
        return None
    return lodtensor


<<<<<<< HEAD
def _rebuild_tensor(cls, lodtensor, metadata):
    if cls == paddle.fluid.framework.EagerParamBase:
        tensor = paddle.fluid.framework.EagerParamBase(
            lodtensor.shape(), lodtensor._dtype(), **metadata
        )
        tensor.value().get_tensor()._share_data_with(lodtensor)
    else:
        size, stop_gradient = metadata
        tensor = paddle.fluid.core.eager.Tensor()
=======
def rebuild_tensor(cls, lodtensor, metadata):
    if cls == paddle.fluid.framework.ParamBase:
        tensor = paddle.fluid.framework.ParamBase(lodtensor.shape(),
                                                  lodtensor._dtype(),
                                                  **metadata)
        tensor.value().get_tensor()._share_data_with(lodtensor)
    else:
        size, stop_gradient = metadata
        tensor = paddle.fluid.core.VarBase()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if lodtensor._is_initialized():
            tensor.value().get_tensor()._share_data_with(lodtensor)
        else:
            tensor = paddle.to_tensor([], dtype=lodtensor._dtype())
        tensor.stop_gradient = stop_gradient
    return tensor


<<<<<<< HEAD
def _reduce_tensor(tensor):
=======
def reduce_tensor(tensor):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    lodtensor = tensor.value().get_tensor()

    if not tensor.stop_gradient and not tensor.is_leaf:
        raise RuntimeError(
            "Refusing to serialize non-leaf tensor which not stop_gradient, you can detach it!"
        )
    # TODO: add serializing name and  hooks check
<<<<<<< HEAD
    if (
        tensor.place.is_cpu_place()
        or tensor.place.is_gpu_place()
        or tensor.place.is_cuda_pinned_place()
    ):
        if type(tensor) == paddle.fluid.framework.EagerParamBase:
=======
    if tensor.place.is_cpu_place() or tensor.place.is_gpu_place(
    ) or tensor.place.is_cuda_pinned_place():
        if type(tensor) == paddle.fluid.framework.ParamBase:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            metadata = copy.deepcopy(tensor.__dict__)
        else:
            metadata = (tensor.size, tensor.stop_gradient)

<<<<<<< HEAD
        return (_rebuild_tensor, (type(tensor), lodtensor, metadata))
    else:
        raise ValueError(
            "Only support tensors of CPU/CUDA/CUDAPinned Place, Not support %s for now!"
            % tensor.place
        )


def _rebuild_lodtensor_filename(cls, ipc_name, size, type_idx, dims, lod):
=======
        return (rebuild_tensor, (type(tensor), lodtensor, metadata))
    else:
        raise ValueError(
            "Only support tensors of CPU/CUDA/CUDAPinned Place, Not support %s for now!"
            % tensor.place)


def rebuild_lodtensor_filename(cls, ipc_name, size, type_idx, dims, lod):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    lodtensor = cls._new_shared_filename((ipc_name, size, type_idx, dims, lod))
    lodtensor._shared_decref()
    return lodtensor


<<<<<<< HEAD
def _rebuild_cuda_tensor(
    cls, handle, offset_bytes, size, type_idx, dims, lod, device_idx
):
    cache_tensor = _cuda_from_cache((handle, offset_bytes))
    if cache_tensor is None:
        lodtensor = cls._new_shared_cuda(
            (handle, offset_bytes, size, type_idx, dims, lod, device_idx)
        )
=======
def rebuild_cuda_tensor(cls, handle, offset_bytes, size, type_idx, dims, lod,
                        device_idx):
    cache_tensor = cuda_from_cache((handle, offset_bytes))
    if cache_tensor is None:
        lodtensor = cls._new_shared_cuda(
            (handle, offset_bytes, size, type_idx, dims, lod, device_idx))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # We only cache cuda shared tensor here.
        # The opening cost of cudaIpcMemoryHandle is very high.
        # Since we cache the recived tensor directly,
        # The sender may reallocate the tensor space,
        # you should manualy maintian the lifecycle of ipc tensor
        shared_cache[(handle, offset_bytes)] = lodtensor
    else:
        lodtensor = paddle.fluid.core.LoDTensor()
<<<<<<< HEAD
        lodtensor._share_buffer_with(
            cache_tensor, (size, type_idx, dims, lod, device_idx)
        )
=======
        lodtensor._share_buffer_with(cache_tensor,
                                     (size, type_idx, dims, lod, device_idx))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return lodtensor


<<<<<<< HEAD
def _rebuild_lodtensor_empty(cls):
    # TODO: check if tensor initialized
    # TODO: handle the dtype of empty tensor
    return cls()


def _reduce_lodtensor(lodtensor):
    if (
        lodtensor._place().is_cpu_place()
        or lodtensor._place().is_cuda_pinned_place()
    ):
        for dim in lodtensor.shape():
            if dim == 0:
                # Empty tensors have nothing be mmapped.
                return (_rebuild_lodtensor_empty, (type(lodtensor),))

        # Default use share filename stratege
        metadata = (
            lodtensor._share_filename()
        )  # ipc_name, size, type_idx, dims, lod
        rebuild = _rebuild_lodtensor_filename
=======
def rebuild_lodtensor_empty(cls):
    #TODO: check if tensor initialized
    #TODO: handle the dtype of empty tensor
    return cls()


def reduce_lodtensor(lodtensor):
    if lodtensor._place().is_cpu_place() or lodtensor._place(
    ).is_cuda_pinned_place():
        for dim in lodtensor.shape():
            if dim == 0:
                # Empty tensors have nothing be mmapped.
                return (rebuild_lodtensor_empty, (type(lodtensor), ))

        # Default use share filename stratege
        metadata = lodtensor._share_filename(
        )  # ipc_name, size, type_idx, dims, lod
        rebuild = rebuild_lodtensor_filename
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        lodtensor._shared_incref()
        # TODO, maintain reference for lodtensor
        # TODO: support file_discriptor stratege
    elif lodtensor._place().is_gpu_place():
        metadata = lodtensor._share_cuda()
<<<<<<< HEAD
        rebuild = _rebuild_cuda_tensor
    else:
        raise RuntimeError("We only support pass cpu/gpu lodtensor for now!")

    return (rebuild, (type(lodtensor),) + metadata)
=======
        rebuild = rebuild_cuda_tensor
    else:
        raise RuntimeError("We only support pass cpu/gpu lodtensor for now!")

    return (rebuild, (type(lodtensor), ) + metadata)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def init_reductions():
    if not _supported_check():
        return

<<<<<<< HEAD
    ForkingPickler.register(paddle.Tensor, _reduce_tensor)
    ForkingPickler.register(paddle.fluid.core.eager.Tensor, _reduce_tensor)
    ForkingPickler.register(
        paddle.fluid.framework.EagerParamBase, _reduce_tensor
    )
    ForkingPickler.register(paddle.fluid.core.LoDTensor, _reduce_lodtensor)
=======
    ForkingPickler.register(paddle.Tensor, reduce_tensor)
    ForkingPickler.register(paddle.fluid.core.VarBase, reduce_tensor)
    ForkingPickler.register(paddle.fluid.framework.ParamBase, reduce_tensor)
    ForkingPickler.register(paddle.fluid.core.LoDTensor, reduce_lodtensor)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
