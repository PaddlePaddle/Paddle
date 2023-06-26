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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/string/string_helper.h"

namespace phi {
namespace funcs {

template <typename T>
static const std::vector<T> &ToVector(const std::vector<T> &vec) {
  return vec;
}

template <typename T>
static std::vector<T> ToVector(const T *x,
                               size_t n,
                               const phi::Place &place UNUSED) {
#ifdef __NVCC__
  if (place.GetType() == phi::AllocationType::GPU) {
    using CopyT = typename std::
        conditional<std::is_same<T, bool>::value, uint8_t, T>::type;
    std::vector<CopyT> cpu_x(n);
    auto *dev_ctx = static_cast<phi::GPUContext *>(
        phi::DeviceContextPool::Instance().Get(place));
    memory_utils::Copy(phi::CPUPlace(),
                       cpu_x.data(),
                       place,
                       x,
                       n * sizeof(T),
                       dev_ctx->stream());
    dev_ctx->Wait();
    return std::vector<T>(cpu_x.data(), cpu_x.data() + n);
  }
#endif
  return std::vector<T>(x, x + n);
}

template <typename T>
static std::vector<T> ToVector(const DenseTensor &src) {
  if (!src.IsInitialized()) {
    return {};
  }
  return ToVector(src.template data<T>(), src.numel(), src.place());
}

template <typename... Args>
static std::string FlattenToString(Args &&...args) {
  const auto &vec = ToVector(std::forward<Args>(args)...);
  return "[" + paddle::string::join_strings(vec, ',') + "]";
}

}  // namespace funcs
}  // namespace phi
