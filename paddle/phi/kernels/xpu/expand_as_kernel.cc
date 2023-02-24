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

#include "paddle/phi/kernels/expand_as_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#define MAX_RANK_SUPPORTED 6

namespace phi {

template <typename Context, typename T>
void ExpandAs(const Context& context,
              const DenseTensor& x,
              const std::vector<int>& target_shape,
              DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto in_dims = x.dims();
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  auto diff = target_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);

  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    PADDLE_ENFORCE_NE(target_shape[i],
                      0,
                      phi::errors::InvalidArgument(
                          "The value of target shape cannot be zero."));
    if (vec_in_dims[i] != 1) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i],
          target_shape[i],
          phi::errors::InvalidArgument(
              "The value (%d) of the non-singleton dimension does not match"
              " the corresponding value (%d) in "
              "target tensor for expand_as_v2 op.",
              vec_in_dims[i],
              target_shape[i]));
    }
  }
  phi::DDim out_dims = phi::make_ddim(target_shape);
  out->Resize(out_dims);
  context.template Alloc<T>(out);
  auto& x_shape = vec_in_dims;
  auto out_shape = phi::vectorize<int>(out_dims);

  int r = XPU_SUCCESS;

  if (std::is_same<T, bool>::value) {
    auto x_data = reinterpret_cast<const int8_t*>(x.data<T>());
    auto out_data = reinterpret_cast<int8_t*>(out->data<T>());
    r = xpu::broadcast<int8_t>(
        context.x_context(), x_data, out_data, x_shape, out_shape);
  } else {
    auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
    auto out_data = reinterpret_cast<XPUType*>(out->data<T>());
    r = xpu::broadcast<XPUType>(
        context.x_context(), x_data, out_data, x_shape, out_shape);
  }
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      phi::errors::External("XPU API(broadcast) return wrong "
                            "value[%d %s] in ExpandAsV2XPUKernel.",
                            r,
                            XPUAPIErrorMsg[r]));
}

template <typename T, typename Context>
void ExpandAsKernel(const Context& ctx,
                    const DenseTensor& x,
                    const paddle::optional<DenseTensor>& y,
                    const std::vector<int>& target_shape,
                    DenseTensor* out) {
  auto rank = x.dims().size();
  auto target_rank = target_shape.size();
  PADDLE_ENFORCE_GE(target_rank,
                    rank,
                    phi::errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be greater than or equal to "
                        "the rank (%d) of the input 'x'.",
                        target_rank,
                        rank));
  PADDLE_ENFORCE_GE(
      rank,
      1,
      phi::errors::InvalidArgument("The rank (%d) of the input 'x' for "
                                   "expand_as_v2 op must be positive.",
                                   rank));
  PADDLE_ENFORCE_LE(target_rank,
                    MAX_RANK_SUPPORTED,
                    phi::errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be less than or equal to %d.",
                        target_rank,
                        MAX_RANK_SUPPORTED));
  ExpandAs<Context, T>(ctx, x, target_shape, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(expand_as,
                   XPU,
                   ALL_LAYOUT,
                   phi::ExpandAsKernel,
                   float,
                   phi::dtype::float16,
                   bool,
                   int,
                   int64_t) {}
