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
const Type*
Type::Get<false /*is_unsupported*/, true /*is_tensor*/, TargetType::kHost,
          PrecisionType::kFloat, DataLayoutType::kNCHW>() {
  static TensorFp32NCHWTy x(TargetType::kHost);
  return &x;
}

template <>
const Type* Type::Get<UnsupportedTy>(TargetType target) {
  return Get<false, false, TargetType::kHost, PrecisionType::kFloat,
             DataLayoutType::kNCHW>();
}

template <TargetType Target>
TensorListAnyTy* GetTensorListAnyTy() {
  static TensorListAnyTy x(Target);
  return &x;
}
template <TargetType Target>
TensorAnyTy* GetTensorAnyTy() {
  static TensorAnyTy x(Target);
  return &x;
}

template <>
const Type* Type::Get<TensorListAnyTy>(TargetType target) {
  switch (target) {
    case TargetType::kHost:
      return GetTensorListAnyTy<TARGET(kHost)>();
    case TargetType::kCUDA:
      return GetTensorListAnyTy<TARGET(kCUDA)>();
    case TargetType::kX86:
      return GetTensorListAnyTy<TARGET(kX86)>();
    default:
      LOG(FATAL) << "unsupported type";
  }
}

template <>
const Type* Type::Get<TensorAnyTy>(TargetType target) {
  switch (target) {
    case TargetType::kHost:
      return GetTensorAnyTy<TARGET(kHost)>();
    case TargetType::kCUDA:
      return GetTensorAnyTy<TARGET(kCUDA)>();
    case TargetType::kX86:
      return GetTensorAnyTy<TARGET(kX86)>();
    default:
      LOG(FATAL) << "unsupported type";
  }
}

template <TargetType Target>
const Type* GetTensorFp32NCHWTy() {
  static TensorFp32NCHWTy x(Target);
  return &x;
}

template <>
const Type* Type::Get<TensorFp32NCHWTy>(TargetType target) {
  switch (target) {
    case TARGET(kHost):
      return GetTensorFp32NCHWTy<TARGET(kHost)>();
    case TARGET(kCUDA):
      return GetTensorFp32NCHWTy<TARGET(kCUDA)>();
    case TARGET(kX86):
      return GetTensorFp32NCHWTy<TARGET(kX86)>();
    default:
      LOG(FATAL) << "unsupported target Type " << TargetToStr(target);
  }
  return nullptr;
}

const Type* LookupType(DataTypeBase::ID type_id, bool is_unknown,
                       bool is_tensor, Place place) {
  using id_t = DataTypeBase::ID;
  switch (type_id) {
    case id_t::Tensor_Any:
      return Type::Get<TensorAnyTy>(place.target);
    case id_t::Tensor_Fp32_NCHW:
      return Type::Get<TensorFp32NCHWTy>(place.target);
    case id_t::TensorList_Any:
      return Type::Get<TensorListAnyTy>(place.target);
    default:
      LOG(FATAL) << "unsupported type";
  }
  return nullptr;
}

// ------------------------- end GetType specification ------------------------

}  // namespace lite
}  // namespace paddle
