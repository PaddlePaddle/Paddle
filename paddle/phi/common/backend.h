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

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/common/place.h"

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
  MLU,  // MLU currently does not exist at the same time as CUDA

  // the third library backend
  MKLDNN,
  GPUDNN,  // cuDNN and hipDNN

  // paddle kernel primitives backend
  KPS,

  IPU,

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
   * PD_REGISTER_KERNEL_FOR_ALL_BACKEND
   *
   * Based on this design pattern, the dtype and layout also have the same
   * requirements, this cause we need to define a series of macros
   *
   * PD_REGISTER_KERNEL_FOR_ALL_DTYPE
   * PD_REGISTER_KERNEL_FOR_ALL_LAYOUT
   * PD_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_LAYOUT
   * PD_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_DTYPE
   * PD_REGISTER_KERNEL_FOR_ALL_LAYOUT_AND_DTYPE
   * PD_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_LAYOUT_AND_DTYPE
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
    case Backend::MLU:
      os << "MLU";
      break;
    case Backend::MKLDNN:
      os << "MKLDNN";
      break;
    case Backend::GPUDNN:
      os << "GPUDNN";
      break;
    case Backend::KPS:
      os << "KPS";
      break;
    case Backend::IPU:
      os << "IPU";
      break;
    default: {
      size_t device_type_id_ = static_cast<size_t>(backend) -
                               static_cast<size_t>(Backend::NUM_BACKENDS);
      std::string device_type = phi::GetGlobalDeviceType(device_type_id_);
      if (!device_type.empty()) {
        os << device_type;
      } else {
        PD_THROW(
            "Invalid enum backend type `", static_cast<int>(backend), "`.");
      }
    }
  }
  return os;
}

inline Backend StringToBackend(const char* backend_cstr) {
  std::string s(backend_cstr);
  if (s == std::string("Undefined")) {
    return Backend::UNDEFINED;
  }
  if (s == std::string("CPU")) {
    return Backend::CPU;
  } else if (s == std::string("GPU")) {
    return Backend::GPU;
  } else if (s == std::string("XPU")) {
    return Backend::XPU;
  } else if (s == std::string("NPU")) {
    return Backend::NPU;
  } else if (s == std::string("MLU")) {
    return Backend::MLU;
  } else if (s == std::string("MKLDNN")) {
    return Backend::MKLDNN;
  } else if (s == std::string("GPUDNN")) {
    return Backend::GPUDNN;
  } else if (s == std::string("KPS")) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    // NOTE(chenweihang) KPS is not yet a complete backend, and it still needs
    // to be converted
    // to GPU in the GPU environment
    return Backend::GPU;
#else
    return Backend::KPS;
#endif
  } else if (s == std::string("IPU")) {
    return Backend::IPU;
  } else {
    return static_cast<Backend>(static_cast<size_t>(Backend::NUM_BACKENDS) +
                                phi::GetOrRegisterGlobalDeviceTypeId(s));
  }
}

}  // namespace experimental
}  // namespace paddle

namespace phi {
using Backend = paddle::experimental::Backend;
}
