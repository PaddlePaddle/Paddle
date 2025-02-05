// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

enum class BoxCodeType { kEncodeCenterSize = 0, kDecodeCenterSize = 1 };

inline BoxCodeType GetBoxCodeType(const std::string &type) {
  PADDLE_ENFORCE_EQ(
      (type == "encode_center_size") || (type == "decode_center_size"),
      true,
      common::errors::InvalidArgument(
          "The 'code_type' attribute in BoxCoder"
          " must be 'encode_center_size' or 'decode_center_size'. "
          "But received 'code_type' is %s",
          type));
  if (type == "encode_center_size") {
    return BoxCodeType::kEncodeCenterSize;
  } else {
    return BoxCodeType::kDecodeCenterSize;
  }
}

}  // namespace funcs
}  // namespace phi
