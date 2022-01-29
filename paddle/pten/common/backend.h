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
  UNDEFINED = 0,

  // basic kernel backend
  CPU,

  // various acceleration devices' backends
  GPU,
  XPU,  // XPU currently does not exist at the same time as CUDA
  NPU,  // NPU currently does not exist at the same time as CUDA

  // the third library backend
  MKLDNN,
  CUDNN,

  // end of backend types
  NUM_BACKENDS,

  /**
   * [ Why we need ALL in baisc kernel key member? ]
   *
   * For Tensor, ALL represents an illegal Backend, but for Kernel, some
   * kernels may be device-independent by nature, such as reshape; and when
   * and some kernels are also device-independent when implemented based on
   * primitive API.
   *
   * In this case, we need to provide a more concise registration method,
   * instead of registering the kernels for each device with almost
   * repetitive code, we need one registration covers all situations,
   * so if we provide the ALL field with Register the kernel in this statement.
   *
   * Of course, we have also considered solving this problem through different
   * named macros, for example, if we define
   *
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND
   *
   * Based on this design pattern, the dtype and layout also have the same
   * requirements, this cause we need to define a series of macros
   *
   * PT_REGISTER_KERNEL_FOR_ALL_DTYPE
   * PT_REGISTER_KERNEL_FOR_ALL_LAYOUT
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_LAYOUT
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_DTYPE
   * PT_REGISTER_KERNEL_FOR_ALL_LAYOUT_AND_DTYPE
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_LAYOUT_AND_DTYPE
   *
   * It makes the system of registering macros more complicated, we think
   * this is not a simple design, so we still adopt the design of providing
   * the ALL field.
   *
   * Note: ALL_BACKEND only used for Kernel registration and selection
   */
  ALL_BACKEND = UNDEFINED,
};

inline std::ostream& operator<<(std::ostream& os, Backend backend) {
  switch (backend) {
    case Backend::UNDEFINED:
      os << "Undefined";
      break;
    case Backend::CPU:
      os << "CPU";
      break;
    case Backend::GPU:
      os << "GPU";
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
