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

#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ExpandKernel(const Context& ctx,
                  const DenseTensor& x,
                  const IntArray& shape,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto in_dims = x.dims();
  auto expand_shape = shape.GetData();
  auto vec_in_dims = common::vectorize<int64_t>(in_dims);
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  auto final_expand_shape = vec_in_dims;
  bool has_zero_dim = false;
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    if (i < diff) {
      PADDLE_ENFORCE_GE(
          expand_shape[i],
          0,
          common::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand_v2 op.",
              expand_shape[i]));
      if (expand_shape[i] == 0) has_zero_dim = true;
      final_expand_shape[i] = expand_shape[i];
    } else if (expand_shape[i] == -1) {
      final_expand_shape[i] = vec_in_dims[i];
    } else if (expand_shape[i] == 0) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i] == 1 || vec_in_dims[i] == expand_shape[i],
          true,
          common::errors::InvalidArgument(
              "The %d-th dimension of input tensor (%d) must match or be "
              "broadcastable to the corresponding dimension (%d) in shape.",
              i,
              vec_in_dims[i],
              expand_shape[i]));
      final_expand_shape[i] = 0;
      has_zero_dim = true;
    } else if (expand_shape[i] > 0) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i] == 1 || vec_in_dims[i] == expand_shape[i],
          true,
          common::errors::InvalidArgument(
              "The %d-th dimension of input tensor (%d) must match or be "
              "broadcastable to the corresponding dimension (%d) in shape.",
              i,
              vec_in_dims[i],
              expand_shape[i]));
      final_expand_shape[i] = expand_shape[i];
    }
  }

  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      common::errors::InvalidArgument(
          "The rank of the input 'X' for expand_v2_npu op must be positive, "
          "but the value received is %d.",
          rank));
  auto shape_size = final_expand_shape.size();
  PADDLE_ENFORCE_GE(
      shape_size,
      rank,
      common::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2_npu op must "
          "be "
          "greater than or equal to the rank (%d) of the input 'X'.",
          shape_size,
          rank));

  DDim out_dims = common::make_ddim(final_expand_shape);
  out->Resize(out_dims);
  ctx.template Alloc<T>(out);
  if (has_zero_dim) {
    return;
  }
  auto& x_shape = vec_in_dims;
  auto out_shape = common::vectorize<int64_t>(out_dims);
  if (shape_size == 0) {
    x_shape = {1};
    out_shape = {1};
  }

  int r = XPU_SUCCESS;
  if (std::is_same<T, bool>::value) {
    auto x_data = reinterpret_cast<const int8_t*>(x.data<T>());
    auto out_data = reinterpret_cast<int8_t*>(out->data<T>());
    r = xpu::broadcast<int8_t>(
        ctx.x_context(), x_data, out_data, x_shape, out_shape);
  } else {
    auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
    auto out_data = reinterpret_cast<XPUType*>(out->data<T>());
    r = xpu::broadcast<XPUType>(
        ctx.x_context(), x_data, out_data, x_shape, out_shape);
  }
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      common::errors::External("XPU API(broadcast) return wrong "
                               "value[%d %s] in ExpandV2XPUKernel.",
                               r,
                               XPUAPIErrorMsg[r]));
}
}  // namespace phi

PD_REGISTER_KERNEL(expand,
                   XPU,
                   ALL_LAYOUT,
                   phi::ExpandKernel,
                   double,
                   float,
                   phi::dtype::float16,
                   bool,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
