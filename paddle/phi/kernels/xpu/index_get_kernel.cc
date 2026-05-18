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

#include "paddle/phi/kernels/index_get_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/xpu/index_put_xpu_utils.h"

namespace phi {
template <typename T, typename Context>
void IndexGetKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices,
                    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));

  std::vector<DenseTensor> tmp_args;
  std::vector<const DenseTensor*> int_indices =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices.empty()) {
    // All bool indices are all-false → output is zero-size with trailing dims
    int64_t effective_num = 0;
    for (const auto* idx : indices) {
      if (idx->dtype() == DataType::BOOL) {
        effective_num += idx->dims().size();
      } else {
        effective_num += 1;
      }
    }
    std::vector<int64_t> out_shape;
    out_shape.push_back(0);
    for (int64_t i = effective_num; i < x.dims().size(); ++i) {
      out_shape.push_back(x.dims()[i]);
    }
    out->Resize(common::make_ddim(out_shape));
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto bd_dim = funcs::BroadCastTensorsDims(int_indices);

  // Stack broadcast indices into [..., num_indices] for gather_nd
  DenseTensor stacked_indices(DataType::INT64);
  XPUDealWithIndices<Context>(dev_ctx, int_indices, bd_dim, &stacked_indices);

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));

  auto x_shape = vectorize<int64_t>(x.dims());
  auto index_shape = vectorize<int64_t>(stacked_indices.dims());
  if (index_shape.size() == 1) {
    index_shape.insert(index_shape.begin(), 1);
  }
  xpu::VectorParam<int64_t> x_vec = {
      x_shape.data(), static_cast<int64_t>(x_shape.size()), nullptr};

  int ret = xpu::gather_nd<XPUType, int64_t>(dev_ctx.x_context(),
                                             x_data,
                                             stacked_indices.data<int64_t>(),
                                             out_data,
                                             x_vec,
                                             index_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather_nd");

  if (dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_get,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexGetKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   bool,
                   phi::float16,
                   phi::bfloat16) {}
