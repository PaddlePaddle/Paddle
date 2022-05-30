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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
constexpr int64_t kNoPadding = -1;

template <typename InT, typename OutT>
static std::vector<OutT> CopyIdsToVector(const DenseTensor &ids) {
  auto numel = ids.numel();
  const auto *src = ids.data<InT>();
  std::vector<OutT> ret(numel);
  if (std::is_same<InT, OutT>::value) {
    std::memcpy(ret.data(), src, numel * sizeof(InT));
  } else {
    for (decltype(numel) i = 0; i < numel; ++i) {
      ret[i] = src[i];
    }
  }
  return ret;
}

}  // namespace phi
