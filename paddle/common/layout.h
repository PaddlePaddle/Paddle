/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <cctype>
#include "paddle/common/exception.h"

namespace common {

// Note: The original design of paddle DataLayout is confusing.
// It contains two levels of "layout", one is the data layout
// at the Tensor level, including Dense, Sparse, etc., and the other
// is the format at the data level, including NHWC, NCHW, etc.,
// these should belong to the concept of "data format".
// The concepts of these two levels are mixed into an enumeration class,
// which leads to some strange execution scheduling logic.
// It needs to be refactored in the future.
// In order to maintain compatibility, we still use the design of the
// original framework here.

// Note: Here the DataLayout is public api for external users, the prefix `k`
// maybe confuse users, so we use all uppercase names

enum class DataLayout {
  UNDEFINED = 0,
  // TODO(chenweihang): keep ANY for compatibility, remove it later
  ANY = UNDEFINED,
  NHWC,
  NCHW,
  NCDHW,
  NDHWC,
  ONEDNN,
  SPARSE_COO,
  SPARSE_CSR,
  PSTRING_UNION,
  STRIDED,

  NUM_DATA_LAYOUTS,

  // See Note [ Why we need ALL in basic kernel key member? ]
  ALL_LAYOUT = UNDEFINED,

  // Note: Unify phi DataLayout and fluid::framework::DataLayout,
  // for compatible with fluid DataLayout, here need prefix `k`

  // Note: The original `kAnyLayout (enum value 2)` is a strange design.
  // `kAnyLayout` originally cannot represent any kind of Layout,
  // at the same time, it can also represent any Layout.
  // Strictly, it means "default" or "undefined" layout,
  // and should not be mixed with other meaningful layouts

  kAnyLayout = ANY,
  kNHWC = NHWC,
  kNCHW = NCHW,
  kMKLDNN = ONEDNN,  // all layouts supported by ONEDNN internally
  kNDHWC = NDHWC,
  kNCDHW = NCDHW,
};

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
  } else if (s == "NDHWC") {
    return DataLayout::kNDHWC;
  } else if (s == "PSTRING_UNION") {
    return DataLayout::PSTRING_UNION;
  } else if (s == "NCDHW") {
    return DataLayout::kNCDHW;
  } else if (s == "STRIDED") {
    return DataLayout::STRIDED;
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
      return "ONEDNN";
    case DataLayout::SPARSE_COO:
      return "SPARSE_COO";
    case DataLayout::SPARSE_CSR:
      return "SPARSE_CSR";
    case DataLayout::kNDHWC:
      return "NDHWC";
    case DataLayout::kNCDHW:
      return "NCDHW";
    case DataLayout::PSTRING_UNION:
      return "PSTRING_UNION";
    case DataLayout::STRIDED:
      return "STRIDED";
    default:
      PD_THROW("Unknown Data Layout type ", static_cast<int>(layout), ".");
  }
}

inline std::ostream& operator<<(std::ostream& os, DataLayout layout) {
  os << DataLayoutToString(layout);
  return os;
}

}  // namespace common

namespace pir {
using DataLayout = common::DataLayout;
}

namespace phi {
using DataLayout = common::DataLayout;
}

namespace paddle {
// In order to be compatible with the original custom operator Tensor interface
using DataLayout = common::DataLayout;

}  // namespace paddle
