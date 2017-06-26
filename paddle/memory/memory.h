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

#include "paddle/frameowork/place.h"

namespace paddle {
namespace memory {

template <typename paddle::framework::Place>
void* Alloc(Place, size_t);
template <typename paddle::framework::Place>
void Free(Place, void*);
template <typename paddle::framework::Place>
size_t Used(Place);

// Staging memory means "pinned" host memory that can be mapped into
// the CUDA memory space and accessed by the device rapidly.  Don't
// allocate too much staging memory; otherwise system performance will
// degrade because the OS cannot find enough swap memory space.
void* AllocStaging(CPUPlace, size_t);
void* FreeStaging(CPUPlace, size_t);

}  // namespace memory
}  // namespace paddle
