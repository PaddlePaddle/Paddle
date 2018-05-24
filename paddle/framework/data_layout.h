/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <iostream>
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

enum DataLayout {
  kNHWC = 0,
  kNCHW = 1,
  kAnyLayout = 2,
};

inline DataLayout StringToDataLayout(const std::string& str) {
  if (str == "NHWC" || str == "nhwc") {
    return DataLayout::kNHWC;
  } else if (str == "NCHW" || str == "nchw") {
    return DataLayout::kNCHW;
  } else {
    PADDLE_THROW("Unknown storage order string: %s", str);
  }
}

inline std::string DataLayoutToString(const DataLayout& data_layout) {
  switch (data_layout) {
    case kNHWC:
      return "NHWC";
    case kNCHW:
      return "NCHW";
    case kAnyLayout:
      return "ANY_LAYOUT";
    default:
      PADDLE_THROW("unknown DataLayou %d", data_layout);
  }
}

inline std::ostream& operator<<(std::ostream& out, DataLayout l) {
  out << DataLayoutToString(l);
  return out;
}

}  // namespace framework
}  // namespace paddle
