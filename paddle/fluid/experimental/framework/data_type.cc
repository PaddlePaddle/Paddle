/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cstddef>

#include "glog/logging.h"

#include "paddle/fluid/experimental/framework/data_type.h"

namespace paddle {
namespace experimental {
namespace framework {

size_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::UINT8:
    case DataType::INT8:
      return 1;
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
      return 2;
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return 4;
    case DataType::FLOAT64:
    case DataType::INT64:
    case DataType::UINT64:
      return 8;
    case DataType::INVALID:
      return 0;
  }
  LOG(FATAL);
}

}  // namespace framework
}  // namespace experimental
}  // namespace paddle
