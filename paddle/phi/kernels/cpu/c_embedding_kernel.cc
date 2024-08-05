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

#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename TIds, typename TData>
void GetIdsEmbedding(const TIds* ids,
                     size_t ids_len,
                     int64_t start_idx,
                     const TData* table,
                     int64_t height,
                     int64_t width,
                     TData* out) {
  for (size_t i = 0; i < ids_len; i++) {
    TIds id = ids[i];
    int64_t local = id - start_idx;

    if (local >= 0 && local < height) {
      memcpy(out + i * width, table + local * width, width * sizeof(TData));
    } else {
      memset(out + i * width, 0, width * sizeof(TData));
    }
  }
}

template <typename T, typename Context>
void CEmbeddingKernel(const Context& ctx,
                      const DenseTensor& w,
                      const DenseTensor& ids,
                      int64_t start_index,
                      int64_t vocab_size,
                      DenseTensor* out) {
  VLOG(10) << "table_dims:" << w.dims();
  const T* table_data = w.data<T>();
  T* output_data = ctx.template Alloc<T>(out);

  const int64_t height = w.dims()[0];
  const int64_t width = w.dims()[1];

  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32) {
    GetIdsEmbedding(ids.data<int32_t>(),
                    ids.numel(),
                    start_index,
                    table_data,
                    height,
                    width,
                    output_data);
  } else if (index_type == phi::DataType::INT64) {
    GetIdsEmbedding(ids.data<int64_t>(),
                    ids.numel(),
                    start_index,
                    table_data,
                    height,
                    width,
                    output_data);
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "CPU c_embedding ids only support int32 or int64."));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(c_embedding,
                   CPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
