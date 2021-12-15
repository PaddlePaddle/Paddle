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

#include "paddle/pten/api/ext/exception.h"
namespace paddle {
namespace experimental {

enum class DataLayout {
  UNDEFINED = 0,
  // TODO(chenweihang): keep ANY for compatibility, remove it later
  ANY = UNDEFINED,
  NHWC,
  NCHW,
  MKLDNN,
  NUM_DATA_LAYOUTS,
  // See Note [ Why we need ALL in baisc kernel key member? ]
  ALL_LAYOUT = UNDEFINED,
};

inline std::ostream& operator<<(std::ostream& os, DataLayout layout) {
  switch (layout) {
    case DataLayout::UNDEFINED:
      os << "Undefined";
      break;
    case DataLayout::NHWC:
      os << "NHWC";
      break;
    case DataLayout::NCHW:
      os << "NCHW";
      break;
    case DataLayout::MKLDNN:
      os << "MKLDNN";
      break;
    default:
      PD_THROW(
          "Invalid enum data layout type `", static_cast<int>(layout), "`.");
  }
  return os;
}

}  // namespace experimental
}  // namespace paddle

namespace pten {
using DataLayout = paddle::experimental::DataLayout;
}
