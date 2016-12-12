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
#include <cmath>
#include "Storage.h"

namespace paddle {

/**
 * Calculate the actual allocation size according to the required size.
 */
MemoryHandle::MemoryHandle(size_t size) : size_(size), buf_(nullptr) {
  if (size_ <= 256) {
    // Memory allocation in cuda is always aligned to at least 256 bytes.
    // In many cases it is 512 bytes.
    allocSize_ = 256;
  } else if (size_ <= 512) {
    allocSize_ = 512;
  } else if (size_ <= (1 << 16)) {
    // Allocate multiple of 1024 bytes.
    allocSize_ = (size + 1023) & ~(1023);
  } else {
    allocSize_ = size_;
  }
}

GpuMemoryHandle::GpuMemoryHandle(size_t size) : MemoryHandle(size) {
  CHECK(size != 0) << " allocate 0 bytes";
  deviceId_ = hl_get_device();
  allocator_ = StorageEngine::singleton()->getGpuAllocator(deviceId_);
  buf_ = allocator_->alloc(allocSize_);
}

GpuMemoryHandle::~GpuMemoryHandle() { allocator_->free(buf_, allocSize_); }

CpuMemoryHandle::CpuMemoryHandle(size_t size) : MemoryHandle(size) {
  CHECK(size != 0) << " allocate 0 bytes";
  allocator_ = StorageEngine::singleton()->getCpuAllocator();
  buf_ = allocator_->alloc(allocSize_);
}

CpuMemoryHandle::~CpuMemoryHandle() { allocator_->free(buf_, allocSize_); }

}  // namespace paddle
