// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {

template <typename T>
static const std::vector<T> &ToVector(const std::vector<T> &vec) {
  return vec;
}

template <typename T>
static std::vector<T> ToVector(const T *x, size_t n,
                               const platform::Place &place) {
#ifdef __NVCC__
  if (platform::is_gpu_place(place)) {
    using CopyT = typename std::conditional<std::is_same<T, bool>::value,
                                            uint8_t, T>::type;
    std::vector<CopyT> cpu_x(n);
    auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    memory::Copy(platform::CPUPlace(), cpu_x.data(), place, x, n * sizeof(T),
                 dev_ctx->stream());
    dev_ctx->Wait();
    return std::vector<T>(cpu_x.data(), cpu_x.data() + n);
  }
#endif
  return std::vector<T>(x, x + n);
}

template <typename T>
static std::vector<T> ToVector(const framework::Tensor &src) {
  if (!src.IsInitialized()) {
    return {};
  }
  return ToVector(src.template data<T>(), src.numel(), src.place());
}

template <typename... Args>
static std::string FlattenToString(Args &&... args) {
  const auto &vec = ToVector(std::forward<Args>(args)...);
  return "[" + string::join_strings(vec, ',') + "]";
}

}  // namespace operators
}  // namespace paddle
