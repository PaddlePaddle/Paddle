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

namespace paddle {
namespace framework {

inline DataType ToDataType(std::type_index type) {
  if (typeid(float).hash_code() == type.hash_code()) {
    return DataType::FP32;
  } else if (typeid(double).hash_code() == type.hash_code()) {
    return DataType::FP64;
  } else if (typeid(int).hash_code() == type.hash_code()) {
    return DataType::INT32;
  } else {
    PADDLE_THROW("Not supported");
    return static_cast<DataType>(-1);
  }
}

}  // namespace framework
}  // namespace paddle
