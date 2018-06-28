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
#include <string>
#include <typeindex>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace framework {

extern proto::VarType::Type ToDataType(std::type_index type);
extern std::type_index ToTypeIndex(proto::VarType::Type type);

template <typename Visitor>
inline void VisitDataType(proto::VarType::Type type, Visitor visitor) {
  switch (type) {
    case proto::VarType::FP16:
      visitor.template operator()<platform::float16>();
      break;
    case proto::VarType::FP32:
      visitor.template operator()<float>();
      break;
    case proto::VarType::FP64:
      visitor.template operator()<double>();
      break;
    case proto::VarType::INT32:
      visitor.template operator()<int>();
      break;
    case proto::VarType::INT64:
      visitor.template operator()<int64_t>();
      break;
    case proto::VarType::BOOL:
      visitor.template operator()<bool>();
      break;
    case proto::VarType::UINT8:
      visitor.template operator()<uint8_t>();
      break;
    case proto::VarType::INT16:
      visitor.template operator()<int16_t>();
      break;
    default:
      PADDLE_THROW("Not supported %d", type);
  }
}

extern std::string DataTypeToString(const proto::VarType::Type type);
extern size_t SizeOfType(std::type_index type);
inline std::ostream& operator<<(std::ostream& out,
                                const proto::VarType::Type& type) {
  out << DataTypeToString(type);
  return out;
}
}  // namespace framework
}  // namespace paddle
