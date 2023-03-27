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

#include <vector>
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
template <typename T, typename Context>
T** GetDevicePointerArray(const Context& ctx,
                          const std::vector<const DenseTensor*>& indices_v) {
  std::vector<const T*> h_indices_v(indices_v.size());
  for (int i = 0; i < indices_v.size(); ++i) {
    h_indices_v[i] = indices_v[i]->data<T>();
  }
  auto d_indices_data = paddle::memory::Alloc(
      ctx.GetPlace(),
      h_indices_v.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  paddle::memory::Copy(ctx.GetPlace(),
                       d_indices_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(h_indices_v.data()),
                       h_indices_v.size() * sizeof(T*),
                       ctx.stream());
  return reinterpret_cast<T**>(d_indices_data->ptr());
}

template <typename T, typename Context>
void IndexPutGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices_v,
                        const DenseTensor& value,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad,
                        DenseTensor* value_grad);

}  // namespace phi
