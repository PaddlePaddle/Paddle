// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/c_embedding_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CEmbeddingKernel(const Context& dev_ctx,
                      const DenseTensor& w,
                      const DenseTensor& ids,
                      int64_t start_index,
                      int64_t vocab_size,
                      DenseTensor* out) {
  const T* table_data = w.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  using XPUType = typename XPUTypeTrait<T>::Type;

  const int64_t height = w.dims()[0];
  const int64_t width = w.dims()[1];

  // int embedding(Context* ctx, const T* x, const TID* indices, T* y, int xm,
  // int n, int ym, int padding_idx, TID start_index = 0);

  // xm: table height: number of entries of table.
  // n: embedding dim: number of float value within single entry.
  // ym: number of elements of input ids.
  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32) {
    int r = xpu::paddle_embedding(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(table_data),
                                  ids.data<int32_t>(),
                                  reinterpret_cast<XPUType*>(output_data),
                                  height,
                                  width,
                                  ids.numel(),
                                  -1,
                                  static_cast<int32_t>(start_index));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_embedding");
  } else if (index_type == phi::DataType::INT64) {
    int r = xpu::paddle_embedding(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(table_data),
                                  ids.data<int64_t>(),
                                  reinterpret_cast<XPUType*>(output_data),
                                  height,
                                  width,
                                  ids.numel(),
                                  -1,
                                  static_cast<int64_t>(start_index));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_embedding");
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "XPU c_embedding ids only support int32 or int64."));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(c_embedding,
                   XPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
