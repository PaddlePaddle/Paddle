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
#include "polaris.h"

namespace paddle {
namespace memory {
namespace detail {

/**
 * \brief SystemAllocator is the parent class of CPUAllocator and GPUAllocator.
 *        A BuddyAllocator object uses a SystemAllocator* pointing to the
 *        underlying system allocator.
 */
class FPGAAllocator {
 public:
  FPGAAllocator(int fpga_id);
  ~FPGAAllocator();
  int GetFPGAId() { return _fpga_id; }
  void* Alloc(size_t size);
  void Free(void* p);
  size_t Used();
 private:
  int _fpga_id;
  PolarisContext* _ctxt;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
