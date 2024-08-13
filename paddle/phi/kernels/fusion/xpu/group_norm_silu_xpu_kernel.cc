// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void GroupNormalizeSiluXPUKernel(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& scale,
                                 const DenseTensor& bias,
                                 int groups,
                                 float epsilon,
                                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* in_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* scale_data = reinterpret_cast<const float*>(scale.data<float>());
  auto* bias_data = reinterpret_cast<const float*>(bias.data<float>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  int n = static_cast<int>(x.dims()[0]);
  int c = static_cast<int>(x.dims()[1]);
  int h = static_cast<int>(x.dims()[2]);
  int w = static_cast<int>(x.dims()[3]);

  int r = xpu::group_norm_silu_fusion<XPUType>(ctx.x_context(),
                                               in_data,
                                               out_data,
                                               n,
                                               c,
                                               h,
                                               w,
                                               groups,
                                               epsilon,
                                               scale_data,
                                               bias_data,
                                               nullptr,
                                               nullptr,
                                               true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "group_norm_silu_fusion");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(group_norm_silu_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::GroupNormalizeSiluXPUKernel,
                   float,
                   phi::dtype::float16) {}
