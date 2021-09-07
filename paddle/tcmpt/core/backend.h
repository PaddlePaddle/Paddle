/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <ostream>
namespace pt {

/**
 * [ Why need Backend? ]
 *
 * Backend not only means place. Backend is a superset of place.
 *
 * Place cannot indicate the difference in calculation methods on the device,
 * but in order to make the boundary of the kernel clearer and the function
 * more specific, we need to distinguish the calculation method.
 *
 * Such as the kernel for CUDA device, it is native CUDA kernel, or kernel
 * by calling CUDNN library.
 */
enum class Backend {
  kUndef = 0,
  kCPU,
  kCUDA,
  kCUDAPinned,  // TODO(chenweihang): need to be removed
  kHIP,         // TODO(chenweihang): hip is not need now
  kXPU,
  kNPU,
  kNPUPinned,  // TODO(chenweihang): need to be removed
  kMKLDNN,
  kCUDNN,
  kNumBackends,
};

std::ostream& operator<<(std::ostream& os, Backend backend);

}  // namespace pt
