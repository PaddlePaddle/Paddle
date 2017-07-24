/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "MemoryHandle.h"

#include "glog/logging.h"

#include <cmath>

#include "paddle/memory/memory.h"
#include "paddle/platform/place.h"

namespace paddle {

/**
 * Calculate the actual allocation size according to the required size.
 */
MemoryHandle::MemoryHandle(size_t size) : size_(size), buf_(nullptr) {}

#ifndef PADDLE_ONLY_CPU

GpuMemoryHandle::GpuMemoryHandle(size_t size) : MemoryHandle(size) {
  CHECK(size != 0) << " allocate 0 bytes";
  deviceId_ = paddle::platform::GetCurrentDeviceId();
  paddle::platform::GPUPlace gpu_place(deviceId_);
  buf_ = paddle::memory::Alloc(gpu_place, size);
}

GpuMemoryHandle::~GpuMemoryHandle() {
  paddle::platform::GPUPlace gpu_place(deviceId_);
  paddle::memory::Free(gpu_place, buf_);
}

#endif  // PADDLE_ONLY_CPU

CpuMemoryHandle::CpuMemoryHandle(size_t size) : MemoryHandle(size) {
  CHECK(size != 0) << " allocate 0 bytes";
  paddle::platform::GPUPlace cpu_place;
  buf_ = paddle::memory::Alloc(cpu_place, size);
}

CpuMemoryHandle::~CpuMemoryHandle() {
  paddle::platform::CPUPlace cpu_place;
  paddle::memory::Free(cpu_place, buf_);
}

}  // namespace paddle
