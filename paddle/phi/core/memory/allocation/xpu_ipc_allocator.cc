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

#include "paddle/phi/core/memory/allocation/xpu_ipc_allocator.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstdlib>
#include <random>
#include <string>

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"

namespace paddle::memory::allocation {

namespace {
// Mutex to protect IPC handle cache.
std::mutex ipc_mutex_;
// Cache mapping from handle string to a weak pointer of the opened IPC memory.
std::unordered_map<std::string, std::weak_ptr<void>> ipc_handle_to_baseptr_;
}  // namespace

std::shared_ptr<void> GetIpcBasePtr(std::string handle) {
  std::lock_guard<std::mutex> lock(ipc_mutex_);

  // Get the current device ID.
  int device_id = platform::GetXPUCurrentDeviceId();
  paddle::platform::SetXPUDeviceId(device_id);

  auto iter = ipc_handle_to_baseptr_.find(handle);
  if (iter != ipc_handle_to_baseptr_.end()) {
    if (auto baseptr = iter->second.lock()) {
      return baseptr;
    }
  }
  // The IPC memory handle can only be opened once for the same handle,
  // so we cache the opened pointer.
  void *baseptr = nullptr;
  // Interpret the provided handle string as an XPU IPC memory handle.
  auto ipc_handle =
      reinterpret_cast<const cudaIpcMemHandle_t *>(handle.c_str());
  // PADDLE_ENFORCE_XPU_SUCCESS(cudaIpcOpenMemHandle(&baseptr, *ipc_handle,
  // cudaIpcMemLazyEnablePeerAccess));
  int ret = cudaIpcOpenMemHandle(
      &baseptr, *ipc_handle, cudaIpcMemLazyEnablePeerAccess);
  PADDLE_ENFORCE_XPU_SUCCESS(ret);

  // Create a shared_ptr with a custom deleter that will close the IPC handle.
  auto sp = std::shared_ptr<void>(baseptr, [handle, device_id](void *ptr) {
    platform::XPUDeviceGuard guard(device_id);
    std::lock_guard<std::mutex> lock(ipc_mutex_);
    PADDLE_ENFORCE_XPU_SUCCESS(cudaIpcCloseMemHandle(ptr));
    ipc_handle_to_baseptr_.erase(handle);
    VLOG(6) << "cudaIpcCloseMemHandle for ptr:"
            << "\t" << ptr;
  });
  std::weak_ptr<void> wp = sp;
  ipc_handle_to_baseptr_.insert({handle, wp});

  return sp;
}

XpuIpcAllocation::~XpuIpcAllocation() {
  // Release the underlying IPC resource.
  shared_ptr_.reset();
  VLOG(6) << "tensor deleted cudaIpcCloseMemHandle for ptr:"
          << "\t" << this->ptr();
}

}  // namespace paddle::memory::allocation

#endif  // _WIN32
