/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <ostream>
#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

enum class DataLayout {
  kNHWC = 0,
  kNCHW = 1,
  kAnyLayout = 2,
  kMKLDNN = 3,  // all layouts supported by MKLDNN internally
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
  } else {
    PADDLE_THROW("Unknown storage order string: %s", s);
  }
}

inline std::string DataLayoutToString(const DataLayout& data_layout) {
  switch (data_layout) {
    case DataLayout::kNHWC:
      return "NHWC";
    case DataLayout::kNCHW:
      return "NCHW";
    case DataLayout::kAnyLayout:
      return "ANY_LAYOUT";
    case DataLayout::kMKLDNN:
      return "MKLDNNLAYOUT";
    default:
      PADDLE_THROW("unknown DataLayout %d", data_layout);
  }
}

inline std::ostream& operator<<(std::ostream& out, const DataLayout& l) {
  out << DataLayoutToString(l);
  return out;
}

}  // namespace framework
}  // namespace paddle
