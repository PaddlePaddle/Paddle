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

// Note: Here the DataLayout is public api for external users, the prefix `k`
// maybe confuse users, so we use all uppercase names
enum class DataLayout {
  UNDEFINED = 0,
  // TODO(chenweihang): keep ANY for compatibility, remove it later
  ANY = UNDEFINED,
  NHWC,
  NCHW,
  MKLDNN,
  SPARSE_COO,
  SPARSE_CSR,
  NUM_DATA_LAYOUTS,
  // See Note [ Why we need ALL in basic kernel key member? ]
  ALL_LAYOUT = UNDEFINED,
  // Note: Unify pten DataLayout and fluid::framework::DataLayout,
  // for compatible with fluid DataLayout, here need prefix `k`
  // Note: The original `kAnyLayout (enum value 2)` is a strange design.
  // `kAnyLayout` originally cannot represent any kind of Layout,
  // at the same time, it can also represent any Layout.
  // Strictly, it means "default" or "undefined" layout,
  // and should not be mixed with other meaningful layouts.
  kAnyLayout = ANY,
  kNHWC = NHWC,
  kNCHW = NCHW,
  kMKLDNN = MKLDNN,  // all layouts supported by MKLDNN internally
};

}  // namespace experimental

// In order to be compatible with the fluid implementation
namespace framework {

using DataLayout = paddle::experimental::DataLayout;

inline DataLayout StringToDataLayout(const std::string& str) {
  std::string s(str);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = toupper(s[i]);
  }

  if (s == "NHWC") {
    return DataLayout::kNHWC;
  } else if (s == "NCHW") {
    return DataLayout::kNCHW;
  } else if (s == "ANYLAYOUT") {
    return DataLayout::kAnyLayout;
  } else if (s == "MKLDNNLAYOUT") {
    return DataLayout::kMKLDNN;
  } else if (s == "SPARSE_COO") {
    return DataLayout::SPARSE_COO;
  } else if (s == "SPARSE_CSR") {
    return DataLayout::SPARSE_CSR;
  } else {
    PD_THROW("Unknown data layout type string: ", s, ".");
  }
}

inline std::string DataLayoutToString(const DataLayout& layout) {
  switch (layout) {
    case DataLayout::kNHWC:
      return "NHWC";
    case DataLayout::kNCHW:
      return "NCHW";
    case DataLayout::kAnyLayout:
      return "Undefined(AnyLayout)";
    case DataLayout::kMKLDNN:
      return "MKLDNN";
    case DataLayout::SPARSE_COO:
      return "SPARSE_COO";
    case DataLayout::SPARSE_CSR:
      return "SPARSE_CSR";
    default:
      PD_THROW("Unknown Data Layout type ", static_cast<int>(layout), ".");
  }
}
}  // namespace framework

namespace experimental {

inline std::ostream& operator<<(std::ostream& os, DataLayout layout) {
  os << framework::DataLayoutToString(layout);
  return os;
}

}  // namespace experimental
}  // namespace paddle

namespace pten {
using DataLayout = paddle::experimental::DataLayout;
}
