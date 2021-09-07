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

#include "paddle/tcmpt/core/layout.h"

namespace pt {

std::ostream& operator<<(std::ostream& os, DataLayout dtype) {
  switch (dtype) {
    case DataLayout::kUndef:
      os << "Undefined";
      break;
    case DataLayout::kAny:
      os << "Any";
      break;
    case DataLayout::kNHWC:
      os << "NHWC";
      break;
    case DataLayout::kNCHW:
      os << "NCHW";
      break;
    case DataLayout::kMKLDNN:
      os << "MKLDNN";
      break;
    default:
      // TODO(chenweihang): change to enforce later
      throw std::runtime_error("Invalid DataLayout type.");
  }
  return os;
}

}  // namespace pt
