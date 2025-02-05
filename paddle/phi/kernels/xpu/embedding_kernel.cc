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

#include "paddle/phi/kernels/embedding_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void EmbeddingKernel(const Context &ctx,
                     const DenseTensor &inputx,
                     const DenseTensor &weight,
                     int64_t padding_idx,
                     DenseTensor *out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto *ids_t = &inputx;  // int
  auto *output_t = out;   // float
  PADDLE_ENFORCE_EQ(
      (std::is_same<Context, XPUContext>::value),
      true,
      common::errors::PreconditionNotMet("Unsupported place! only support "
                                         "xpu place , please check your "
                                         "place."));

  int64_t ids_numel = ids_t->numel();

  auto *table_t = &weight;
  auto &dev_ctx = ctx;

  auto *table = table_t->data<T>();
  auto *output = dev_ctx.template Alloc<T>(output_t);

  PADDLE_ENFORCE_EQ(
      ids_numel <= std::numeric_limits<int32_t>::max(),
      true,
      common::errors::OutOfRange(
          "Number of ids greater than int32_t::max , please check "
          "number of ids in LookupTableV2XPUKernel."));

  int ym = static_cast<int>(ids_numel);

  size_t xm = table_t->dims()[0];
  size_t n = table_t->dims()[1];

  int r;
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  if (ids_t->dtype() == phi::DataType::INT64) {
#ifndef PADDLE_WITH_XPU_PLUGIN
    r = xpu::paddle_embedding<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType *>(table),
        ids_t->data<int64_t>(),
        reinterpret_cast<XPUType *>(output),
        xm,
        n,
        ym,
        padding_idx);
#else
    r = xpu::plugin::fast_embedding<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType *>(table),
        ids_t->data<int64_t>(),
        reinterpret_cast<XPUType *>(output),
        xm,
        n,
        ym,
        padding_idx);
#endif
  } else {
#ifndef PADDLE_WITH_XPU_PLUGIN
    int64_t *ids_tt = RAII_GUARD.alloc_l3_or_gm<int64_t>(ids_t->numel());
    r = xpu::cast<int32_t, int64_t>(
        ctx.x_context(), ids_t->data<int>(), ids_tt, ids_t->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    const int64_t *ids = reinterpret_cast<const int64_t *>(ids_tt);
    r = xpu::paddle_embedding<XPUType>(dev_ctx.x_context(),
                                       reinterpret_cast<const XPUType *>(table),
                                       ids,
                                       reinterpret_cast<XPUType *>(output),
                                       xm,
                                       n,
                                       ym,
                                       padding_idx);
#else
    r = xpu::plugin::fast_embedding<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType *>(table),
        ids_t->data<int>(),
        reinterpret_cast<XPUType *>(output),
        xm,
        n,
        ym,
        padding_idx);
#endif
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_embedding");
}

}  // namespace phi

PD_REGISTER_KERNEL(embedding,
                   XPU,
                   ALL_LAYOUT,
                   phi::EmbeddingKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
