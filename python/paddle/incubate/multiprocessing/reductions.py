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

import copy
import multiprocessing

# TODO: check the hooks of tensor
# TODO: check serializing named tensor
# TODO: check influence on autograd
import sys
import threading
from collections import OrderedDict
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork

import paddle


def _supported_check():
    if sys.platform != "linux":
        # warnings.warn("`paddle.multiprocessing` only support linux for now, "
        #               " import this will not take any effect !")

        return False

    return True


class _LRUSharedCache(OrderedDict):
    def __init__(self):
        self.limit = 128
        self._after_fork()
        register_after_fork(self, _LRUSharedCache._after_fork)

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


shared_cache = _LRUSharedCache()


def _cuda_from_cache(key):
    lodtensor = shared_cache.get(key)
    if lodtensor is None:
        return None
    return lodtensor


def _rebuild_tensor(cls, lodtensor, metadata):
    if cls == paddle.base.framework.EagerParamBase:
        tensor = paddle.base.framework.EagerParamBase(
            lodtensor.shape(), lodtensor._dtype(), **metadata
        )
        tensor.value().get_tensor()._share_data_with(lodtensor)
    else:
        size, stop_gradient = metadata
        tensor = paddle.base.core.eager.Tensor()
        if lodtensor._is_initialized():
            tensor.get_tensor()._share_data_with(lodtensor)
        else:
            tensor = paddle.to_tensor([], dtype=lodtensor._dtype())
        tensor.stop_gradient = stop_gradient
    return tensor


def _rebuild_vmm_tensor(
    cls, fd_or_dup, offset_bytes, size, type_idx, dims, lod, device_idx
):
    print("============== _rebuild_vmm_tensor ===========")
    # 与 CPU 共享文件的 rebuild 一样，DupFd 需要先 detach 成真正的 int FD
    if hasattr(fd_or_dup, "detach"):
        fd = fd_or_dup.detach()
    else:
        fd = int(fd_or_dup)
    try:
        import os

        os.fstat(fd)  # EBADF 会在这里抛出，便于定位
    except Exception as e:
        raise ValueError(f"VMM meta fd invalid before import: fd={fd}, err={e}")
    # 直接用你在 pybind 里加的 _new_shared_vmm 重建
    lodtensor = cls._new_shared_vmm(
        (fd, offset_bytes, size, type_idx, dims, lod, device_idx)
    )
    return lodtensor


def _reduce_tensor(tensor):
    print("========== debug _reduce_tensor")
    lodtensor = tensor.get_tensor()

    if not tensor.stop_gradient and not tensor.is_leaf:
        raise RuntimeError(
            "Refusing to serialize non-leaf tensor which not stop_gradient, you can detach it!"
        )
    # TODO: add serializing name and  hooks check
    if (
        tensor.place.is_cpu_place()
        or tensor.place.is_gpu_place()
        or tensor.place.is_cuda_pinned_place()
        or tensor.place.is_xpu_place()
    ):
        if type(tensor) == paddle.base.framework.EagerParamBase:
            metadata = copy.deepcopy(tensor.__dict__)
        else:
            metadata = (tensor.size, tensor.stop_gradient)

        return (_rebuild_tensor, (type(tensor), lodtensor, metadata))
    else:
        raise ValueError(
            f"Only support tensors of CPU/CUDA/CUDAPinned Place, Not support {tensor.place} for now!"
        )


def _rebuild_lodtensor_filename(
    cls,
    ipc_name,
    shared_fd,
    size,
    type_idx,
    dims,
    lod,
    dataloader_use_file_descriptor,
):
    lodtensor = cls._new_shared_filename(
        (
            ipc_name,
            shared_fd,
            size,
            type_idx,
            dims,
            lod,
            dataloader_use_file_descriptor,
        )
    )
    lodtensor._shared_decref()
    return lodtensor


def _rebuild_lodtensor_filedescriptor(
    cls,
    ipc_name,
    shared_fd,
    size,
    type_idx,
    dims,
    lod,
    dataloader_use_file_descriptor,
):
    shared_fd = shared_fd.detach()
    lodtensor = cls._new_shared_filename(
        (
            ipc_name,
            shared_fd,
            size,
            type_idx,
            dims,
            lod,
            dataloader_use_file_descriptor,
        )
    )
    lodtensor._shared_decref()
    return lodtensor


def _rebuild_cuda_tensor(
    cls, handle, offset_bytes, size, type_idx, dims, lod, device_idx
):
    print("=========== debug _rebuild_cuda_tensor ========")
    cache_tensor = _cuda_from_cache((handle, offset_bytes))
    if cache_tensor is None:
        lodtensor = cls._new_shared_cuda(
            (handle, offset_bytes, size, type_idx, dims, lod, device_idx)
        )
        # We only cache cuda shared tensor here.
        # The opening cost of cudaIpcMemoryHandle is very high.
        # Since we cache the received tensor directly,
        # The sender may reallocate the tensor space,
        # you should manually maintain the lifecycle of ipc tensor
        shared_cache[(handle, offset_bytes)] = lodtensor
    else:
        lodtensor = paddle.base.core.DenseTensor()
        lodtensor._share_buffer_with(
            cache_tensor, (size, type_idx, dims, lod, device_idx)
        )

    return lodtensor


def _rebuild_xpu_tensor(
    cls, handle, offset_bytes, size, type_idx, dims, lod, device_idx
):
    cache_tensor = _cuda_from_cache((handle, offset_bytes))
    if cache_tensor is None:
        lodtensor = cls._new_shared_xpu(
            (handle, offset_bytes, size, type_idx, dims, lod, device_idx)
        )
        # We only cache cuda shared tensor here.
        # The opening cost of cudaIpcMemoryHandle is very high.
        # Since we cache the received tensor directly,
        # The sender may reallocate the tensor space,
        # you should manually maintain the lifecycle of ipc tensor
        shared_cache[(handle, offset_bytes)] = lodtensor
    else:
        lodtensor = paddle.base.core.DenseTensor()
        lodtensor._share_buffer_with(
            cache_tensor, (size, type_idx, dims, lod, device_idx)
        )

    return lodtensor


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
                # Empty tensors have nothing be mapped.
                return (_rebuild_lodtensor_empty, (type(lodtensor),))
        dataloader_use_file_descriptor = paddle.base.core.globals()[
            "FLAGS_dataloader_use_file_descriptor"
        ]
        # Default use share filename strategy
        metadata = lodtensor._share_filename(
            dataloader_use_file_descriptor
        )  # ipc_name, fd, size, type_idx, dims, lod

        if dataloader_use_file_descriptor:
            metalist = list(metadata)
            metalist[1] = multiprocessing.reduction.DupFd(metalist[1])
            metadata = tuple(metalist)
            rebuild = _rebuild_lodtensor_filedescriptor
        else:
            rebuild = _rebuild_lodtensor_filename
        lodtensor._shared_incref()
        # TODO, maintain reference for lodtensor
    elif lodtensor._place().is_gpu_place():
        """
        prev_id = paddle.base.core.get_cuda_current_device_id()
        cur_id = lodtensor._place().gpu_device_id()
        if prev_id != cur_id:
            paddle.base.core.set_cuda_current_device_id(cur_id)
        try:
            metadata = lodtensor._share_cuda()
        finally:
            if prev_id != cur_id:
                paddle.base.core.set_cuda_current_device_id(prev_id)
        """

        prev_id = paddle.base.core.get_cuda_current_device_id()
        cur_id = lodtensor._place().gpu_device_id()
        if prev_id != cur_id:
            paddle.base.core.set_cuda_current_device_id(cur_id)
        try:
            if paddle.get_flags('FLAGS_use_virtual_memory_auto_growth')[
                'FLAGS_use_virtual_memory_auto_growth'
            ]:
                # print("=========== vmm lodtensor: ", lodtensor)
                fd, offset, size, type_idx, dims, lod, device = (
                    lodtensor._share_vmm()
                )
                if not hasattr(fd, "detach"):
                    from multiprocessing.reduction import DupFd

                    # 像 CPU 的 file-descriptor 路径一样，用 DupFd 传 fd
                    fd = DupFd(fd)
                metadata = (fd, offset, size, type_idx, dims, lod, device)
                rebuild = _rebuild_vmm_tensor
            else:
                print("=========== cuda lodtensor: ", lodtensor)
                metadata = lodtensor._share_cuda()
                rebuild = _rebuild_cuda_tensor
        finally:
            if prev_id != cur_id:
                paddle.base.core.set_cuda_current_device_id(prev_id)
    elif lodtensor._place().is_xpu_place():
        metadata = lodtensor._share_xpu()
        rebuild = _rebuild_xpu_tensor
    else:
        raise RuntimeError(
            "We only support pass cpu/gpu/xpu lodtensor for now!"
        )

    return (rebuild, (type(lodtensor), *metadata))


def init_reductions() -> None:
    if not _supported_check():
        return

    ForkingPickler.register(paddle.Tensor, _reduce_tensor)
    ForkingPickler.register(paddle.base.core.eager.Tensor, _reduce_tensor)
    ForkingPickler.register(
        paddle.base.framework.EagerParamBase, _reduce_tensor
    )
    ForkingPickler.register(paddle.base.core.DenseTensor, _reduce_lodtensor)
