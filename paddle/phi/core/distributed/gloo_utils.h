// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <gloo/math.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/types.h>

#include <climits>
#include <memory>
#include <string>

#include "glog/logging.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/reduce_helper.h"

namespace phi {
namespace distributed {

// data preparation
#ifdef _WIN32
#define GENERATE_FUNC(type, func, ...)       \
  switch (type) {                            \
    case phi::DataType::FLOAT32:             \
      func<float>(__VA_ARGS__);              \
      break;                                 \
    case phi::DataType::FLOAT64:             \
      func<double>(__VA_ARGS__);             \
      break;                                 \
    case phi::DataType::FLOAT16:             \
      func<gloo::float16>(__VA_ARGS__);      \
      break;                                 \
    case phi::DataType::INT32:               \
      func<int32_t>(__VA_ARGS__);            \
      break;                                 \
    case phi::DataType::INT64:               \
      func<int64_t>(__VA_ARGS__);            \
      break;                                 \
    default:                                 \
      VLOG(0) << "Error: Unknown DataType."; \
      exit(-1);                              \
  }
#define HOST_NAME_MAX 256
#else
#define GENERATE_FUNC(type, func, args...)   \
  switch (type) {                            \
    case phi::DataType::FLOAT32:             \
      func<float>(args);                     \
      break;                                 \
    case phi::DataType::FLOAT64:             \
      func<double>(args);                    \
      break;                                 \
    case phi::DataType::FLOAT16:             \
      func<gloo::float16>(args);             \
      break;                                 \
    case phi::DataType::INT32:               \
      func<int32_t>(args);                   \
      break;                                 \
    case phi::DataType::INT64:               \
      func<int64_t>(args);                   \
      break;                                 \
    case phi::DataType::INT8:                \
      func<int8_t>(args);                    \
      break;                                 \
    case phi::DataType::UINT8:               \
      func<uint8_t>(args);                   \
      break;                                 \
    case phi::DataType::BOOL:                \
      func<bool>(args);                      \
      break;                                 \
    case phi::DataType::BFLOAT16:            \
      func<phi::dtype::bfloat16>(args);      \
      break;                                 \
    default:                                 \
      VLOG(0) << "Error: Unknown DataType."; \
      exit(-1);                              \
  }
#endif

template <typename T, typename P>
void SetOutput(P* opts, phi::DenseTensor* tensor) {
  opts->setOutput(reinterpret_cast<T*>(tensor->data()), tensor->numel());
}

template <typename T, typename P>
void SetInput(P* opts, const phi::DenseTensor& tensor) {
  // gloo only support mutable data input
  opts->setInput(reinterpret_cast<T*>(const_cast<void*>(tensor.data())),
                 tensor.numel());
}

template <typename T, typename P>
void SetReduceFunc(P* opts, int reduce_type) {
  // gloo only support mutable data input
  switch (reduce_type) {
    case kRedSum:
      opts->setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::sum<T>));
      break;
    case kRedMax:
      opts->setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::max<T>));
      break;
    case kRedMin:
      opts->setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::min<T>));
      break;
    case kRedProd:
      opts->setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::product<T>));
      break;
    default:
      PADDLE_THROW(
          errors::InvalidArgument("Invalid reduce type: %d.", reduce_type));
  }
}

// env preparation
std::shared_ptr<gloo::transport::Device> CreateGlooDevice();

}  // namespace distributed
}  // namespace phi
