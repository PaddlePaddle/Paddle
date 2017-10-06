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

#pragma once

#include <stddef.h>  // for size_t

namespace paddle {
namespace memory {
namespace detail {

/**
 * \brief SystemAllocator is the parent class of CPUAllocator and GPUAllocator.
 *        A BuddyAllocator object uses a SystemAllocator* pointing to the
 *        underlying system allocator.
 */
class SystemAllocator {
 public:
  virtual ~SystemAllocator() {}
  virtual void* Alloc(size_t& index, size_t size) = 0;
  virtual void Free(void* p, size_t size, size_t index) = 0;
  virtual bool UseGpu() const = 0;
};

class CPUAllocator : public SystemAllocator {
 public:
  virtual void* Alloc(size_t& index, size_t size);
  virtual void Free(void* p, size_t size, size_t index);
  virtual bool UseGpu() const;
};

#ifdef PADDLE_WITH_CUDA
class GPUAllocator : public SystemAllocator {
 public:
  virtual void* Alloc(size_t& index, size_t size);
  virtual void Free(void* p, size_t size, size_t index);
  virtual bool UseGpu() const;

 private:
  size_t gpu_alloc_size_ = 0;
  size_t fallback_alloc_size_ = 0;
};
#endif  // PADDLE_ONLY_CPU

}  // namespace detail
}  // namespace memory
}  // namespace paddle
