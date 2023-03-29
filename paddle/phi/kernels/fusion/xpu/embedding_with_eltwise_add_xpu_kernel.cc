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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void EmbeddingWithEltwiseAddXpuKernel(
    const Context& ctx,
    const std::vector<const DenseTensor*>& ids,
    const std::vector<const DenseTensor*>& tables,
    int64_t padding_idx,
    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto& id_dims = ids[0]->dims();
  int idx_len = id_dims[0] * id_dims[1];
  int emb_layer_num = ids.size();
  int embed_dim = tables[0]->dims()[1];
  std::vector<int> table_lens_cpu;
  std::vector<const float*> arg_tables;
  for (auto* table : tables) {
    auto& table_dims = table->dims();
    PADDLE_ENFORCE_EQ(
        table_dims.size(),
        2,
        errors::InvalidArgument(
            "The table_dims size [%d] should be equal 2.",
            table_dims.size())); /* shape like [table_len, embed_dim] */
    PADDLE_ENFORCE_EQ(
        table_dims[1],
        embed_dim,
        errors::InvalidArgument(
            "Every embed_dim [%d] should be equal the first one [%d].",
            table_dims[1],
            embed_dim));
    table_lens_cpu.push_back(table_dims[0]);
    if (std::is_same<T, phi::dtype::float16>::value) {
      DenseTensor table_data_fp32_t;
      ctx.template Alloc<float>(&table_data_fp32_t,
                                table->numel() * sizeof(float));
      int r = xpu::cast<XPUType, float>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(table->data<T>()),
          table_data_fp32_t.data<float>(),
          table->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      arg_tables.push_back(table_data_fp32_t.data<float>());
    } else {
      arg_tables.push_back(table->data<float>());
    }
  }
  std::vector<std::vector<int>> int_idx(emb_layer_num,
                                        std::vector<int>(idx_len, 0));
  std::vector<xpu::VectorParam<int>> arg_ids;
  for (int i = 0; i < emb_layer_num; i++) {
    PADDLE_ENFORCE_EQ(
        ids[i]->dtype() == phi::DataType::INT64 ||
            ids[i]->dtype() == phi::DataType::INT32,
        true,
        errors::InvalidArgument(
            "The data type of ids should be int64 or int32, but got %s.",
            ids[i]->dtype()));
    for (int j = 0; j < idx_len; j++) {
      if (ids[i]->dtype() == phi::DataType::INT64) {
        int_idx[i][j] = static_cast<int>(ids[i]->data<int64_t>()[j]);
      } else if (ids[i]->dtype() == phi::DataType::INT32) {
        int_idx[i][j] = ids[i]->data<int>()[j];
      }
    }
    arg_ids.push_back(
        xpu::VectorParam<int>{int_idx[i].data(), idx_len, nullptr});
  }

  ctx.template Alloc<T>(out);
  if (std::is_same<T, phi::dtype::float16>::value) {
    DenseTensor out_fp32_t;
    ctx.template Alloc<float>(&out_fp32_t, out->numel() * sizeof(float));
    int r = xpu::multi_embedding_fusion<float, float, int>(
        ctx.x_context(),
        arg_tables, /* tables */
        out_fp32_t.data<float>(),
        arg_ids,
        table_lens_cpu,
        embed_dim,
        std::vector<float>(table_lens_cpu.size(), 1.0f),
        std::vector<int>(table_lens_cpu.size(), padding_idx));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_with_eltwise_add_xpu");

    r = xpu::cast(ctx.x_context(),
                  out_fp32_t.data<float>(),
                  reinterpret_cast<XPUTypeFP16*>(out->data<T>()),
                  out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  } else {
    int r = xpu::multi_embedding_fusion<float, float, int>(
        ctx.x_context(),
        arg_tables, /* tables */
        out->data<float>(),
        arg_ids,
        table_lens_cpu,
        embed_dim,
        std::vector<float>(table_lens_cpu.size(), 1.0f),
        std::vector<int>(table_lens_cpu.size(), padding_idx));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_with_eltwise_add_xpu");
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(embedding_with_eltwise_add_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::EmbeddingWithEltwiseAddXpuKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
}
