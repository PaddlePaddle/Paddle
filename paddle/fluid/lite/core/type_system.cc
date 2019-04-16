// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {

// ------------------------- GetType specification ----------------------------
template <>
const Type*
Type::Get<false /*is_unsupported*/, false /*is_tensor*/, TargetType::kHost,
          PrecisionType::kFloat, DataLayoutType::kNCHW>() {
  static UnsupportedTy x;
  return &x;
}

template <>
const Type*
Type::Get<false /*is_unsupported*/, true /*is_tensor*/, TargetType::kX86,
          PrecisionType::kFloat, DataLayoutType::kNCHW>() {
  static TensorFp32NCHWTy x(TargetType::kX86);
  return &x;
}

template <>
const Type* Type::Get<UnsupportedTy>(TargetType target) {
  return Get<false, false, TargetType::kHost, PrecisionType::kFloat,
             DataLayoutType::kNCHW>();
}

template <>
const Type* Type::Get<TensorFp32NCHWTy>(TargetType target) {
  switch (target) {
    case TargetType::kX86:
      return Get<false, true, TargetType::kX86, PrecisionType::kFloat,
                 DataLayoutType::kNCHW>();
    default:
      LOG(FATAL) << "unsupported target " << TargetToStr(target);
      return nullptr;
  }
}

// ------------------------- end GetType specification ------------------------

}  // namespace lite
}  // namespace paddle
