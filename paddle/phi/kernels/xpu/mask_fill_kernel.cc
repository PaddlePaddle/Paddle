// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/masked_fill_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void MaskedFillKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& mask,
                      const DenseTensor& value,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (x.numel() == 0 || mask.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto& x_dims = x.dims();
  const auto& mask_dims = mask.dims();

  DDim x_dims_ex = x_dims;
  DDim mask_dims_ex = mask_dims;

  if (x_dims.size() == 0 && mask_dims.size() == 0) {
    x_dims_ex = common::make_ddim({1});
    mask_dims_ex = common::make_ddim({1});
  } else {
    int rank = std::max(x_dims.size(), mask_dims.size());
    x_dims_ex = funcs::ExtendDims2Rank(x_dims, rank);
    mask_dims_ex = funcs::ExtendDims2Rank(mask_dims, rank);
  }

  auto out_dims = funcs::BroadcastTwoDims(x_dims_ex, mask_dims_ex, -1);
  out->Resize(out_dims);
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (out && out->numel() == 0) {
    return;
  }

  const bool* cond_data = mask.data<bool>();
  const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  XPUType* out_xpu = reinterpret_cast<XPUType*>(out_data);

  auto cond_vec = vectorize<int64_t>(mask_dims_ex);
  auto x_vec = vectorize<int64_t>(x_dims_ex);

  auto* ctx = dev_ctx.x_context();
  int r = xpu::SUCCESS;

  DenseTensor value_expand;
  const DenseTensor* value_tensor = &value;
  if (value.dims() != x_dims) {
    auto target = vectorize(x_dims);
    phi::ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(target), &value_expand);
    value_tensor = &value_expand;
  }

  const XPUType* y_data =
      reinterpret_cast<const XPUType*>(value_tensor->data<T>());

  r = xpu::masked_fill(
      ctx, cond_data, x_data, y_data, out_xpu, cond_vec, x_vec);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "masked_fill_tensor");
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_fill,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaskedFillKernel,
                   float,
                   int,
                   int8_t,
                   int64_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
