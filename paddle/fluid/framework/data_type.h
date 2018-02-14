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
#include <typeindex>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

inline proto::VarType::Type ToDataType(std::type_index type) {
  using namespace paddle::framework::proto;
  if (typeid(float).hash_code() == type.hash_code()) {
    return VarType_Type_FP32;
  } else if (typeid(double).hash_code() == type.hash_code()) {
    return VarType_Type_FP64;
  } else if (typeid(int).hash_code() == type.hash_code()) {
    return VarType_Type_INT32;
  } else if (typeid(int64_t).hash_code() == type.hash_code()) {
    return VarType_Type_INT64;
  } else if (typeid(bool).hash_code() == type.hash_code()) {
    return VarType_Type_BOOL;
  } else {
    PADDLE_THROW("Not supported");
  }
}

inline std::type_index ToTypeIndex(proto::VarType::Type type) {
  using namespace paddle::framework::proto;
  switch (type) {
    case VarType_Type_FP32:
      return typeid(float);
    case VarType_Type_FP64:
      return typeid(double);
    case VarType_Type_INT32:
      return typeid(int);
    case VarType_Type_INT64:
      return typeid(int64_t);
    case VarType_Type_BOOL:
      return typeid(bool);
    default:
      PADDLE_THROW("Not support type %d", type);
  }
}

template <typename Visitor>
inline void VisitDataType(proto::VarType::Type type, Visitor visitor) {
  using namespace paddle::framework::proto;
  switch (type) {
    case VarType_Type_FP32:
      visitor.template operator()<float>();
      break;
    case VarType_Type_FP64:
      visitor.template operator()<double>();
      break;
    case VarType_Type_INT32:
      visitor.template operator()<int>();
      break;
    case VarType_Type_INT64:
      visitor.template operator()<int64_t>();
      break;
    case VarType_Type_BOOL:
      visitor.template operator()<bool>();
      break;
    default:
      PADDLE_THROW("Not supported");
  }
}

inline std::string DataTypeToString(const proto::VarType::Type type) {
  using namespace paddle::framework::proto;
  switch (type) {
    case VarType_Type_FP16:
      return "float16";
    case VarType_Type_FP32:
      return "float32";
    case VarType_Type_FP64:
      return "float64";
    case VarType_Type_INT16:
      return "int16";
    case VarType_Type_INT32:
      return "int32";
    case VarType_Type_INT64:
      return "int64";
    case VarType_Type_BOOL:
      return "bool";
    default:
      PADDLE_THROW("Not support type %d", type);
  }
}

inline std::ostream& operator<<(std::ostream& out,
                                const proto::VarType::Type& type) {
  out << DataTypeToString(type);
  return out;
}

}  // namespace framework
}  // namespace paddle
