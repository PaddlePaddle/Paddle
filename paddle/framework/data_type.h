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
#include <typeindex>
#include "paddle/framework/framework.pb.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

inline DataType ToDataType(std::type_index type) {
  if (typeid(float).hash_code() == type.hash_code()) {
    return DataType::FP32;
  } else if (typeid(double).hash_code() == type.hash_code()) {
    return DataType::FP64;
  } else if (typeid(int).hash_code() == type.hash_code()) {
    return DataType::INT32;
  } else if (typeid(int64_t).hash_code() == type.hash_code()) {
    return DataType::INT64;
  } else {
    PADDLE_THROW("Not supported");
  }
}

inline std::type_index ToTypeIndex(DataType type) {
  switch (type) {
    case DataType::FP32:
      return typeid(float);
    case DataType::FP64:
      return typeid(double);
    case DataType::INT32:
      return typeid(int);
    case DataType::INT64:
      return typeid(int64_t);
    default:
      PADDLE_THROW("Not support type %d", type);
  }
}

template <typename Visitor>
inline void VisitDataType(DataType type, Visitor visitor) {
  switch (type) {
    case DataType::FP32:
      visitor.template operator()<float>();
      break;
    case DataType::FP64:
      visitor.template operator()<double>();
      break;
    case DataType::INT32:
      visitor.template operator()<int>();
      break;
    case DataType::INT64:
      visitor.template operator()<int64_t>();
      break;
    default:
      PADDLE_THROW("Not supported");
  }
}

}  // namespace framework
}  // namespace paddle
