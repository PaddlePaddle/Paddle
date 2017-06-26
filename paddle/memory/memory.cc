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

#include "paddle/memory/memory.h"

namespace paddle {
namespace memory {

template <>
void* Alloc<CPUPlace>(CPUPlace, size_t size) {
  return GetCPUBuddyAllocator(false /*non-staging*/)->Alloc(size);
}

void* AllocStaging(CPUPlace, size_t size) {
  return GetCPUBuddyAllocator(true /*staging*/)->Alloc(size);
}

template <>
void* Alloc<GPUPlace>(GPUPlace pl, size_t size) {
  return GetGPUBuddyAllocator(pl.device)->Alloc(size);
}

template <>
void Free<CPUPlace>(CPUPlace, void* p) {
  return GetCPUBuddyAllocator(false /*non-staging*/)->Free(p);
}

void FreeStaging(CPUPlace, void* p) {
  return GetCPUBuddyAllocator(false /*non-staging*/)->Free(p);
}

#ifdef PADDLE_WITH_GPU
template <>
void* Alloc<GPUPlace>(GPUPlace pl, void* p) {
  return GetGPUBuddyAllocator(pl.device)->Free(p);
}

template <>
size_t Used<CPUPlace>(CPUPlace) {
  return GetCPUBuddyAllocator()->Used();
}

template <>
size_t Alloc<GPUPlace>(GPUPlace pl) {
  return GetGPUBuddyAllocator(pl.device)->Used();
}
#endif  // PADDLE_WITH_GPU

}  // namespace memory
}  // namespace paddle
