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

// SystemAllocator is the parent class of CPUAllocator and
// GPUAllocator.  A BuddyAllocator object uses a SystemAllocator*
// pointing to the underlying system allocator.  An alternative to
// this class hierarchy is to pass a system allocator class to
// BuddyAllocator as a template parameter.  This approach makes
// BuddyAllocator a class template, and it's very complicated
// algorithm would make the buddy_allocator.h messy.
class SystemAllocator {
 public:
  virtual ~SystemAllocator() {}
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p, size_t size) = 0;
};

class CPUAllocator : public SystemAllocator {
 public:
  virtual void* Alloc(size_t size);
  virtual void Free(void* p, size_t size);
};

#ifndef PADDLE_ONLY_CPU
class GPUAllocator : public SystemAllocator {
 public:
  virtual void* Alloc(size_t size);
  virtual void Free(void* p, size_t size);
};
#endif  // PADDLE_ONLY_CPU

}  // namespace detail
}  // namespace memory
}  // namespace paddle
