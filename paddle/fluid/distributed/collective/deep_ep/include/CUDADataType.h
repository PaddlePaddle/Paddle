// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>
#include <variant>
#include "paddle/common/overloaded.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/ScalarType.h"
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"

namespace deep_ep::detail {

template <phi::DataType phi_data_type>
struct PhiDataTypeImpl {
  constexpr static phi::DataType value = phi_data_type;
};

using PhiDataType = std::variant<
#define MAKE_PHI_DATA_TYPE_CASE(_, phi_data_type) \
  PhiDataTypeImpl<phi::phi_data_type>,
    PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_CASE)
        PhiDataTypeImpl<phi::DataType::UNDEFINED>
#undef MAKE_PHI_DATA_TYPE_CASE
    >;

inline PhiDataType ScalarTypeToPhiDataType(
    const deep_ep::detail::ScalarType& scalar_type) {
  static std::unordered_map<deep_ep::detail::ScalarType, PhiDataType> map = {
#define MAKE_PHI_DATA_TYPE_CONVERT_CASE(_, phi_data_type) \
  {phi::phi_data_type, PhiDataTypeImpl<phi::phi_data_type>{}},
      PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_CONVERT_CASE)
#undef MAKE_PHI_DATA_TYPE_CONVERT_CASE
          {phi::DataType::UNDEFINED,
           PhiDataTypeImpl<phi::DataType::UNDEFINED>{}},
  };
  const auto iter = map.find(scalar_type);
  if (iter == map.end()) {
    LOG(FATAL) << "unsupported scalar type: " << static_cast<int>(scalar_type);
  }
  return iter->second;
}

inline cudaDataType_t ScalarTypeToCudaDataType(
    const deep_ep::detail::ScalarType& scalar_type) {
  auto phi_data_type = detail::ScalarTypeToPhiDataType(scalar_type);
  auto Converter = ::common::Overloaded{
      [](detail::PhiDataTypeImpl<phi::DataType::PSTRING>) -> cudaDataType_t {
        LOG(FATAL) << "unsupported scalar type: pstring";
        return *(cudaDataType_t*)nullptr;  // NOLINT
      },
      [](detail::PhiDataTypeImpl<phi::DataType::UNDEFINED>) -> cudaDataType_t {
        LOG(FATAL) << "unsupported scalar type: undefined";
        return *(cudaDataType_t*)nullptr;  // NOLINT
      },
      [](auto phi_data_type_impl) -> cudaDataType_t {
        using T = std::decay_t<decltype(phi_data_type_impl)>;
        using CppT = typename phi::DataTypeToCppType<T::value>::type;
        return phi::backends::gpu::ToCudaDataType<CppT>();
      }};
  return std::visit(Converter, phi_data_type);
}

}  // namespace deep_ep::detail
