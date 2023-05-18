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
    const paddle::optional<DenseTensor>& mask,
    int64_t padding_idx,
    DenseTensor* out,
    DenseTensor* seq_lod,
    DenseTensor* max_seq_len) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int emb_dim = tables[0]->dims()[1];
  std::vector<int> table_lens;
  std::vector<const float*> arg_tables;
  for (auto* table : tables) {
    auto& table_dims = table->dims();
    PADDLE_ENFORCE_EQ(
        table_dims.size(),
        2,
        errors::InvalidArgument(
            "The table_dims size [%d] should be equal to 2.",
            table_dims.size())); /* shape like [table_len, emb_dim] */
    PADDLE_ENFORCE_EQ(
        table_dims[1],
        emb_dim,
        errors::InvalidArgument(
            "Every emb_dim [%d] should be equal to the first one [%d].",
            table_dims[1],
            emb_dim));
    table_lens.push_back(table_dims[0]);
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

  int emb_layer_num = ids.size();
  for (int i = 0; i < emb_layer_num; i++) {
    auto id_dtype = ids[i]->dtype();
    PADDLE_ENFORCE(
        id_dtype == phi::DataType::INT64 || id_dtype == phi::DataType::INT32,
        errors::InvalidArgument(
            "The data type of ids should be int64 or int32, but got %s.",
            DataTypeToString(id_dtype)));
  }

  auto& id_dims = ids[0]->dims();
  int batch_size = id_dims[0];
  int max_seq_len_value = id_dims[1];
  ctx.template HostAlloc<int>(max_seq_len)[0] = max_seq_len_value;
  int ids_len = id_dims[0] * id_dims[1];
  std::vector<std::vector<int>> int_ids(emb_layer_num,
                                        std::vector<int>(ids_len, 0));
  std::vector<xpu::VectorParam<int>> arg_ids;
  auto* mask_tensor = mask.get_ptr();
  if (mask_tensor != nullptr) {
    auto mask_dtype = mask_tensor->dtype();
    PADDLE_ENFORCE(
        mask_dtype == phi::DataType::INT64 ||
            mask_dtype == phi::DataType::FLOAT32,
        errors::InvalidArgument(
            "The data type of mask should be int64 or float32, but got %s.",
            DataTypeToString(mask_dtype)));

    int* seq_lod_data = ctx.template HostAlloc<int>(seq_lod);
    seq_lod_data[0] = 0;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      int cur_batch_seq_len = 0;
      for (int seq_idx = 0; seq_idx < max_seq_len_value; seq_idx++) {
        int mask_idx = batch_idx * max_seq_len_value + seq_idx;
        if ((mask_dtype == phi::DataType::INT64 &&
             mask->data<int64_t>()[mask_idx] > 0) ||
            (mask_dtype == phi::DataType::FLOAT32 &&
             fabs(mask->data<float>()[mask_idx]) > 1e-5)) {
          cur_batch_seq_len++;
        } else {
          break;
        }
      }
      PADDLE_ENFORCE_GT(
          cur_batch_seq_len,
          0,
          errors::PreconditionNotMet(
              "cur_batch_seq_len should be greater than 0, but got %d.",
              cur_batch_seq_len));
      seq_lod_data[batch_idx + 1] = seq_lod_data[batch_idx] + cur_batch_seq_len;
    }
    out->Resize({batch_size, seq_lod_data[batch_size + 1], emb_dim});

    for (int i = 0; i < emb_layer_num; i++) {
      if (ids[i]->dtype() == DataType::INT64) {
        auto* ids_data = ids[i]->data<int64_t>();
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
          for (int j = 0;
               j < seq_lod_data[batch_idx + 1] - seq_lod_data[batch_idx];
               j++) {
            int_ids[i][seq_lod_data[batch_idx] + j] =
                ids_data[batch_idx * max_seq_len_value + j];
          }
        }
      } else {
        auto* ids_data = ids[i]->data<int>();
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
          for (int j = 0;
               j < seq_lod_data[batch_idx + 1] - seq_lod_data[batch_idx];
               j++) {
            int_ids[i][seq_lod_data[batch_idx] + j] =
                ids_data[batch_idx * max_seq_len_value + j];
          }
        }
      }
      arg_ids.push_back(
          xpu::VectorParam<int>{int_ids[i].data(), ids_len, nullptr});
    }
  } else {
    for (int i = 0; i < emb_layer_num; i++) {
      for (int j = 0; j < ids_len; j++) {
        if (ids[i]->dtype() == phi::DataType::INT64) {
          int_ids[i][j] = static_cast<int>(ids[i]->data<int64_t>()[j]);
        } else if (ids[i]->dtype() == phi::DataType::INT32) {
          int_ids[i][j] = ids[i]->data<int>()[j];
        }
      }
      arg_ids.push_back(
          xpu::VectorParam<int>{int_ids[i].data(), ids_len, nullptr});
    }
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
        table_lens,
        emb_dim,
        std::vector<float>(table_lens.size(), 1.0f),
        std::vector<int>(table_lens.size(), padding_idx));
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
        table_lens,
        emb_dim,
        std::vector<float>(table_lens.size(), 1.0f),
        std::vector<int>(table_lens.size(), padding_idx));
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
  kernel->InputAt(2).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(1).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(2).SetBackend(phi::Backend::CPU);
}
