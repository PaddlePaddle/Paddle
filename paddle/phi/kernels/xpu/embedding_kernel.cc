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
  auto *ids_t = &inputx;  // int
  auto *output_t = out;   // float
  PADDLE_ENFORCE_EQ(
      (std::is_same<Context, XPUContext>::value),
      true,
      phi::errors::PreconditionNotMet("Unsupported place! only support "
                                      "xpu place , please check your "
                                      "place."));

  int64_t ids_numel = ids_t->numel();

  auto *table_t = &weight;
  auto &dev_ctx = ctx;

  auto *table = table_t->data<T>();
  auto *output = dev_ctx.template Alloc<T>(output_t);

  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  const int64_t *ids;
  if (ids_t->dtype() == phi::DataType::INT64) {
    ids = ids_t->data<int64_t>();
  } else {
    int64_t *ids_tt = RAII_GUARD.alloc_l3_or_gm<int64_t>(ids_t->numel());
    int r = xpu::cast<int32_t, int64_t>(
        ctx.x_context(), ids_t->data<int>(), ids_tt, ids_t->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    ids = reinterpret_cast<const int64_t *>(ids_tt);
  }

  PADDLE_ENFORCE_EQ(
      ids_numel <= std::numeric_limits<int32_t>::max(),
      true,
      phi::errors::OutOfRange(
          "Number of ids greater than int32_t::max , please check "
          "number of ids in LookupTableV2XPUKernel."));

  int ym = static_cast<int>(ids_numel);

  size_t xm = table_t->dims()[0];
  size_t n = table_t->dims()[1];

  int r = xpu::embedding<T, int64_t>(dev_ctx.x_context(),
                                     table,
                                     ids,
                                     output,
                                     xm,
                                     n,
                                     ym,
                                     static_cast<int>(padding_idx));

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding");
}

}  // namespace phi

PD_REGISTER_KERNEL(embedding, XPU, ALL_LAYOUT, phi::EmbeddingKernel, float) {}
