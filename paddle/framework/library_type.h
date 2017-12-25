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

namespace paddle {
namespace framework {

// For more details about the design of LibraryType, Please refer to
// https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/operator_kernel_type.md#library

enum LibraryType { kPlain = 0, kMKLDNN = 1, kCUDNN = 2 };

inline std::string LibraryTypeToString(const LibraryType& library_type) {
  switch (library_type) {
    case kPlain:
      return "PLAIN";
    case kMKLDNN:
      return "MKLDNN";
    case kCUDNN:
      return "CUDNN";
    default:
      PADDLE_THROW("unknown LibraryType %d", library_type);
  }
}

inline std::ostream& operator<<(std::ostream& out, LibraryType l) {
  out << LibraryTypeToString(l);
  return out;
}

}  // namespace
}  // framework
