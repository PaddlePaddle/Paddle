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

#include "paddle/tcmpt/core/backend.h"

namespace pt {

std::ostream& operator<<(std::ostream& os, Backend backend) {
  switch (backend) {
    case Backend::kUndef:
      os << "Undefined";
      break;
    case Backend::kCPU:
      os << "CPU";
      break;
    case Backend::kCUDA:
      os << "CUDA";
      break;
    case Backend::kCUDAPinned:
      os << "CUDAPinned";
      break;
    case Backend::kHIP:
      os << "HIP";
      break;
    case Backend::kXPU:
      os << "XPU";
      break;
    case Backend::kNPU:
      os << "NPU";
      break;
    case Backend::kNPUPinned:
      os << "NPUPinned";
      break;
    case Backend::kMKLDNN:
      os << "MKLDNN";
      break;
    case Backend::kCUDNN:
      os << "CUDNN";
      break;
    default:
      // TODO(chenweihang): change to enforce later
      throw std::runtime_error("Invalid Backend type.");
  }
  return os;
}

}  // namespace pt
