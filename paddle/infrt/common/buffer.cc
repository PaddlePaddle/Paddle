// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/common/buffer.h"

#include <stdarg.h>
#include <stdio.h>

#include <cmath>

namespace infrt {
void Buffer::Resize(uint32_t size) {
  if (size_ > 0) {
    Free();
    size_ = 0;
  }

  if (size_ != size) {
    data_.memory = reinterpret_cast<uint8_t*>(Malloc(size));
    size_ = size;
  }
}

void Buffer::Resize(uint32_t alignment, uint32_t size) {
  if (size_ > 0) {
    Free();
    size_ = 0;
  }

  if (size_ != size) {
    data_.memory = reinterpret_cast<uint8_t*>(AlignedAlloc(alignment, size));
    size_ = size;
  }
}

void Buffer::SetTarget(const infrt::common::Target& target) {
  target_ = target;
  memory_mng_cache_ = MemoryManager::Global().RetrieveSafely(target_.arch);
}

void Buffer::ResizeLazy(uint32_t size) {
  if (size <= size_) return;
  Resize(size);
}

void Buffer::ResizeLazy(uint32_t alignment, uint32_t size) {
  if (size <= size_) return;
  Resize(alignment, size);
}

void Buffer::Resize(uint32_t size, const infrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  Resize(size);
}

void Buffer::Resize(uint32_t alignment,
                    uint32_t size,
                    const infrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  Resize(alignment, size);
}

void Buffer::ResizeLazy(uint32_t size, const infrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  ResizeLazy(size);
}

void Buffer::ResizeLazy(uint32_t alignment,
                        uint32_t size,
                        const infrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  ResizeLazy(alignment, size);
}

}  // namespace infrt
