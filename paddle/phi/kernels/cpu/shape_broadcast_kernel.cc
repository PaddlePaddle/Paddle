/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/utils/array_ref.h"

namespace phi {

template <typename T>
std::vector<T> ComputeBroadcastShape(const paddle::array_ref<T>& large_shape,
                                     const paddle::array_ref<T>& small_shape) {
  PADDLE_ENFORCE_GE(
      large_shape.size(),
      small_shape.size(),
      common::errors::PreconditionNotMet(
          "Size of large_shape is expected to be greater or equal size of "
          "small_shape, but got [%d] >= [%d].",
          large_shape.size(),
          small_shape.size()));
  std::vector<T> output_data;
  output_data.reserve(large_shape.size());
  auto rank_gap = large_shape.size() - small_shape.size();
  for (size_t i = 0; i < rank_gap; ++i) {
    output_data.push_back(large_shape[i]);
  }
  for (size_t i = 0; i < small_shape.size(); ++i) {
    output_data.push_back(std::max(large_shape[i + rank_gap], small_shape[i]));
  }
  return output_data;
}

template <typename T, typename Context>
void ShapeBroadcastKernel(const Context& ctx,
                          const DenseTensor& x_shape,
                          const DenseTensor& y_shape,
                          DenseTensor* out) {
  PADDLE_ENFORCE_EQ(x_shape.dims().size(),
                    1,
                    common::errors::InvalidArgument(
                        "Invalid input tensor. The rank of x_shape "
                        "should be equal 1, but now received [%d].",
                        x_shape.dims().size()));
  PADDLE_ENFORCE_EQ(y_shape.dims().size(),
                    1,
                    common::errors::InvalidArgument(
                        "Invalid input tensor. The rank of y_shape "
                        "should be equal 1, but now received [%d].",
                        y_shape.dims().size()));
  paddle::array_ref<T> x_shape_data(x_shape.data<T>(), x_shape.numel());
  paddle::array_ref<T> y_shape_data(y_shape.data<T>(), y_shape.numel());
  const auto& output_data =
      x_shape_data.size() > y_shape_data.size()
          ? ComputeBroadcastShape(x_shape_data, y_shape_data)
          : ComputeBroadcastShape(y_shape_data, x_shape_data);
  T* out_data = ctx.template HostAlloc<T>(out);
  int64_t out_numel = out->numel();
  for (int i = 0; i < out_numel; ++i) {
    out_data[i] = output_data[i];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(shape_broadcast,
                   CPU,
                   ALL_LAYOUT,
                   phi::ShapeBroadcastKernel,
                   int32_t,
                   int64_t) {}
