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

#include "paddle/phi/kernels/one_hot_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& depth,
                  DenseTensor* out) {
  auto depth_v = depth.to<int>();
  auto out_dims = out->dims();
  if (out_dims[out_dims.size() - 1] == -1) {
    out_dims[out_dims.size() - 1] = depth_v;
    out->Resize(out_dims);
  }
  auto* p_in_data = x.data<T>();
  auto numel = x.numel();
  auto* p_out_data = dev_ctx.template Alloc<float>(out);
  if (numel == 0) return;
  int r = xpu::one_hot<T>(
      dev_ctx.x_context(), p_in_data, p_out_data, numel, depth_v, 1.0, 0.0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "one_hot");
}
}  // namespace phi

PD_REGISTER_KERNEL(one_hot, XPU, ALL_LAYOUT, phi::OneHotKernel, int, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
