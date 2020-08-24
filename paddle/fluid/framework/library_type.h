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
#include <string>

namespace paddle {
namespace framework {

// For more details about the design of LibraryType, Please refer to
// https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/operator_kernel_type.md#library

enum class LibraryType {
  kPlain = 0,
  kMKLDNN = 1,
  kCUDNN = 2,
};

inline std::string LibraryTypeToString(const LibraryType& library_type) {
  switch (library_type) {
    case LibraryType::kPlain:
      return "PLAIN";
    case LibraryType::kMKLDNN:
      return "MKLDNN";
    case LibraryType::kCUDNN:
      return "CUDNN";
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unknown LibraryType code (%d), only supports library type include "
          "PLAIN(0), MKLDNN(1), CUDNN(2).",
          static_cast<int>(library_type)));
  }
}

inline LibraryType StringToLibraryType(const char* ctype) {
  std::string s(ctype);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = toupper(s[i]);
  }
  if (s == std::string("PLAIN")) {
    return LibraryType::kPlain;
  } else if (s == std::string("MKLDNN")) {
    return LibraryType::kMKLDNN;
  } else if (s == std::string("CUDNN")) {
    return LibraryType::kCUDNN;
    // To be compatible with register macro.
    // CPU, CUDA, PLAIN are same library type.
  } else if (s == std::string("CPU")) {
    return LibraryType::kPlain;
  } else if (s == std::string("XPU")) {
    return LibraryType::kPlain;
  } else if (s == std::string("CUDA")) {
    return LibraryType::kPlain;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unknown LibraryType string (%s), only support library type string "
        "include PLAIN, MKLDNN, CUDNN, CPU and CUDA.",
        s.c_str()));
  }
}

inline std::ostream& operator<<(std::ostream& out, LibraryType l) {
  out << LibraryTypeToString(l);
  return out;
}

}  // namespace framework
}  // namespace paddle
