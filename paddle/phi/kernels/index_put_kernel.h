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

inline void GetBroadcastDimsArrays(const DDim& x_dims,
                                   const DDim& y_dims,
                                   int* x_dims_array,
                                   int* y_dims_array,
                                   int* out_dims_array,
                                   const int max_dim,
                                   const int axis) {
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      phi::errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array + axis);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
            y_dims_array[i] <= 1,
        true,
        phi::errors::InvalidArgument(
            "Broadcast dimension mismatch. Operands could "
            "not be broadcast together with the shape of X = [%s] and "
            "the shape of Y = [%s]. Received [%d] in X is not equal to "
            "[%d] in Y at i:%d.",
            x_dims,
            y_dims,
            x_dims_array[i],
            y_dims_array[i],
            i));
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = std::max(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

static DDim BroadcastTwoDims(const DDim& x_dims, const DDim& y_dims, int axis) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);
  return phi::make_ddim(out_dims_array);
}

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
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices_v,
                    const DenseTensor& value,
                    DenseTensor* out);

}  // namespace phi
