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

#include "paddle/phi/kernels/embedding_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& ctx,
                         const DenseTensor& input,
                         const DenseTensor& weight,
                         const DenseTensor& out_grad,
                         int64_t padding_idx,
                         DenseTensor* weight_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  DDim table_dim;
  table_dim = weight.dims();

  auto ids_t = &input;
  auto d_output_t = &out_grad;
  auto d_table_t = weight_grad;

  int64_t ids_numel = ids_t->numel();
  PADDLE_ENFORCE_EQ(
      ids_numel <= std::numeric_limits<int32_t>::max(),
      true,
      phi::errors::OutOfRange(
          "Number of ids greater than int32_t::max , please check "
          "number of ids in LookupTableV2GradXPUKernel."));

  auto& dev_ctx = ctx;
  //   const int64_t* ids_data = ids_t->data<int64_t>();
  //   const T* d_output_data = d_output_t->data<T>();
  //   T* d_table_data = dev_ctx.template Alloc<T>(d_table_t);
  const XPUType* d_output_data =
      reinterpret_cast<const XPUType*>(d_output_t->data<T>());
  XPUType* d_table_data =
      reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(d_table_t));

  auto* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  const int64_t* ids;
  if (input.dtype() == phi::DataType::INT64) {
    ids = ids_t->data<int64_t>();
  } else if (input.dtype() == phi::DataType::INT16) {
    int64_t* ids_tmp = RAII_GUARD.alloc_l3_or_gm<int64_t>(ids_t->numel());
    int r = xpu::cast<int16_t, int64_t>(
        xpu_ctx, ids_t->data<int16_t>(), ids_tmp, ids_t->numel());
    ids = ids_tmp;
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding inputx only support int16, int32 and int64"));
  }

  int xm = d_table_t->dims()[0];
  int ym = static_cast<int>(ids_numel);
  int n = d_table_t->dims()[1];

  int r = xpu::embedding_grad<XPUType, int64_t>(
      xpu_ctx, d_output_data, ids, d_table_data, xm, n, ym, padding_idx);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(embedding_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::EmbeddingGradKernel,
                   float,
                   phi::dtype::float16) {}
