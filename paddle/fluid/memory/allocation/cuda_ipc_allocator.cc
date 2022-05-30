// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _WIN32

#include "paddle/fluid/memory/allocation/cuda_ipc_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <random>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {
std::mutex ipc_mutex_;
std::unordered_map<std::string, std::weak_ptr<void>> ipc_handle_to_baseptr_;
}  // namespace

std::shared_ptr<void> GetIpcBasePtr(std::string handle) {
  std::lock_guard<std::mutex> lock(ipc_mutex_);

  auto iter = ipc_handle_to_baseptr_.find(handle);
  if (iter != ipc_handle_to_baseptr_.end()) {
    auto baseptr = iter->second.lock();
    if (baseptr) return baseptr;
  }
  // The IpcMemHandle can only open once for the same handle,
  // so here we cache it here.
  void *baseptr = nullptr;
  auto ipc_handle =
      reinterpret_cast<const cudaIpcMemHandle_t *>(handle.c_str());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(
      &baseptr, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
  // Close ipc handle on the same device.
  int device_id = platform::GetCurrentDeviceId();
  // Add deleter to close ipc handle.
  auto sp = std::shared_ptr<void>(baseptr, [handle, device_id](void *ptr) {
    platform::CUDADeviceGuard guard(device_id);
    std::lock_guard<std::mutex> lock(ipc_mutex_);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcCloseMemHandle(ptr));
    ipc_handle_to_baseptr_.erase(handle);
    VLOG(6) << "cudaIpcCloseMemHandle for ptr:"
            << "\t" << ptr;
  });
  std::weak_ptr<void> wp = sp;
  ipc_handle_to_baseptr_.insert(iter, {handle, wp});

  return sp;
}

CudaIpcAllocation::~CudaIpcAllocation() {
  shared_ptr_.reset();
  VLOG(6) << "tensor deleted cudaIpcCloseMemHandle for ptr:"
          << "\t" << this->ptr();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
