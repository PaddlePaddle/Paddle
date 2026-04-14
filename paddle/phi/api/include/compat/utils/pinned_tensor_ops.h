// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/Scalar.h>

#include "paddle/common/ddim.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"

namespace compat {

inline bool _PD_IsHostPinnedTensor(const paddle::Tensor& tensor) {
  const auto& place = tensor.place();
  return phi::is_cuda_pinned_place(place) || phi::is_xpu_pinned_place(place);
}

inline void _PD_FillTensorInplace(paddle::Tensor* tensor,
                                  const c10::Scalar& value) {
  if (!_PD_IsHostPinnedTensor(*tensor)) {
    paddle::experimental::fill_(*tensor, value);
    return;
  }

  auto cpu_src = paddle::experimental::full(
      phi::IntArray(common::vectorize<int64_t>(tensor->dims())),
      value,
      tensor->dtype(),
      phi::CPUPlace());
  tensor->copy_(cpu_src, tensor->place(), /*blocking=*/true);
}

}  // namespace compat
