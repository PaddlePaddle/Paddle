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

#include "paddle/pten/api/ext/exception.h"

namespace paddle {
namespace experimental {

/**
 * [ Why need Backend? ]
 *
 * Backend not only means place. Backend is a superset of place.
 *
 * Place cannot indicate the difference in calculation methods on the device,
 * but in order to make the boundary of the kernel clearer and the function
 * more specific, we need to distinguish the calculation method.
 *
 * Such as the kernel for CPU device, it can be a native CPU kernel,
 * or a kernel implemented by MKLDNN library.
 *
 * Note(chenweihang): HIP is not needed now, we can added it if needed
 * in the future
 */
enum class Backend : uint8_t {
  // kernel backend cannot be undefined
  UNDEFINED = 0,

  // basic kernel backend
  CPU,

  // various acceleration devices' backends
  CUDA,
  XPU,  // XPU currently does not exist at the same time as CUDA
  NPU,  // NPU currently does not exist at the same time as CUDA

  // the third library backend
  MKLDNN,
  CUDNN,

  // end of backend types
  NUM_BACKENDS,
};

inline std::ostream& operator<<(std::ostream& os, Backend backend) {
  switch (backend) {
    case Backend::UNDEFINED:
      os << "Undefined";
      break;
    case Backend::CPU:
      os << "CPU";
      break;
    case Backend::CUDA:
      os << "CUDA";
      break;
    case Backend::XPU:
      os << "XPU";
      break;
    case Backend::NPU:
      os << "NPU";
      break;
    case Backend::MKLDNN:
      os << "MKLDNN";
      break;
    case Backend::CUDNN:
      os << "CUDNN";
      break;
    default:
      PD_THROW("Invalid enum backend type `", static_cast<int>(backend), "`.");
  }
  return os;
}

}  // namespace experimental
}  // namespace paddle

namespace pten {
using Backend = paddle::experimental::Backend;
}
