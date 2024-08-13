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

#include "paddle/phi/kernels/c_embedding_grad_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename TIds, typename TData>
void UpdateEmbedding(const TIds* ids,
                     size_t ids_len,
                     int64_t start_idx,
                     TData* table,
                     int64_t height,
                     int64_t width,
                     const TData* out) {
  for (size_t i = 0; i < ids_len; i++) {
    TIds id = ids[i];
    int64_t local = id - start_idx;

    if (local >= 0 && local < height) {
      for (int64_t w = 0; w < width; w++) {
        table[local * width + w] += out[i * width + w];
      }
    }
  }
}

template <typename T, typename Context>
void CEmbeddingGradKernel(const Context& dev_ctx,
                          const DenseTensor& w,
                          const DenseTensor& ids,
                          const DenseTensor& out_grad,
                          int64_t start_index,
                          DenseTensor* w_grad) {
  w_grad->Resize(w.dims());
  T* table_grad_data = dev_ctx.template Alloc<T>(w_grad);

  size_t table_t_mem_size = w.numel() * sizeof(w_grad->dtype());
  size_t table_grad_t_mem_size = w_grad->numel() * sizeof(w_grad->dtype());

  VLOG(10) << "table_dims:" << w.dims()
           << ", table_t memory_size:" << table_t_mem_size
           << ", table_grad_t memory_size:" << table_grad_t_mem_size
           << ", start_index:" << start_index;

  memset(table_grad_data, 0, table_grad_t_mem_size);
  const T* d_output_data = out_grad.data<T>();

  const int64_t height = w.dims()[0];
  const int64_t width = w.dims()[1];

  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32) {
    UpdateEmbedding(ids.data<int32_t>(),
                    ids.numel(),
                    start_index,
                    table_grad_data,
                    height,
                    width,
                    d_output_data);
  } else if (index_type == phi::DataType::INT64) {
    UpdateEmbedding(ids.data<int64_t>(),
                    ids.numel(),
                    start_index,
                    table_grad_data,
                    height,
                    width,
                    d_output_data);
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "CPU c_embedding ids only support int32 or int64."));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(c_embedding_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
