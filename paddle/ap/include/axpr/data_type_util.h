// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/phi/common/data_type.h"

namespace ap::axpr {

inline Result<DataType> GetDataTypeFromPhiDataType(::phi::DataType data_type) {
  static const std::unordered_map<::phi::DataType, DataType> map{
      {::phi::DataType::UNDEFINED, DataType{CppDataType<adt::Undefined>{}}},
#define MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE(cpp_type, enum_type) \
  {::phi::enum_type, DataType{CppDataType<cpp_type>{}}},
      PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE)
#undef MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE
  };
  const auto& iter = map.find(data_type);
  if (iter == map.end()) {
    return adt::errors::KeyError{[&] {
      std::ostringstream ss{};
      ss << "Invalid phi data type. enum value: " << data_type;
      return ss.str();
    }()};
  }
  return iter->second;
}

inline Result<::phi::DataType> GetPhiDataTypeFromDataType(DataType data_type) {
  static const std::unordered_map<DataType, ::phi::DataType> map{
      {DataType{CppDataType<adt::Undefined>{}}, ::phi::DataType::UNDEFINED},
#define MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE(cpp_type, enum_type) \
  {DataType{CppDataType<cpp_type>{}}, ::phi::enum_type},
      PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE)
#undef MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE
  };
  const auto& iter = map.find(data_type);
  if (iter == map.end()) {
    return adt::errors::KeyError{"Invalid axpr data type."};
  }
  return iter->second;
}

}  // namespace ap::axpr
