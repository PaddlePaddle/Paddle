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

namespace {
template <typename T>
void FillSeqLod(int batch_size, int max_seq_len, const T* mask, int* seq_lod) {
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int cur_batch_seq_len = 0;
    for (int seq_idx = 0; seq_idx < max_seq_len; seq_idx++) {
      int mask_idx = batch_idx * max_seq_len + seq_idx;
      if (mask[mask_idx] > 0) {
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
    seq_lod[batch_idx + 1] = seq_lod[batch_idx] + cur_batch_seq_len;
  }
}

template <>
void FillSeqLod<float>(int batch_size,
                       int max_seq_len,
                       const float* mask,
                       int* seq_lod) {
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int cur_batch_seq_len = 0;
    for (int seq_idx = 0; seq_idx < max_seq_len; seq_idx++) {
      int mask_idx = batch_idx * max_seq_len + seq_idx;
      if (fabs(mask[mask_idx]) > 1e-5) {
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
    seq_lod[batch_idx + 1] = seq_lod[batch_idx] + cur_batch_seq_len;
  }
}

template <typename TT, typename TID, typename Context>
void MultiEmbeddingKernel(const Context& ctx,
                          const std::vector<const DenseTensor*>& ids,
                          const std::vector<const DenseTensor*>& tables,
                          const paddle::optional<DenseTensor>& mask,
                          int64_t padding_idx,
                          DenseTensor* out,
                          DenseTensor* seq_lod,
                          DenseTensor* max_seq_len) {
  using XPUType = typename XPUTypeTrait<TT>::Type;
  int64_t emb_dim = tables[0]->dims()[1];
  std::vector<TID> table_lens;
  std::vector<const XPUType*> arg_tables;
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
    arg_tables.push_back(reinterpret_cast<const XPUType*>(table->data<TT>()));
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
  auto* mask_tensor = mask.get_ptr();
  if (mask_tensor != nullptr) {
    max_seq_len->Resize({1});
    ctx.template HostAlloc<int>(max_seq_len)[0] = max_seq_len_value;

    seq_lod->Resize({batch_size + 1});
    int* seq_lod_data = ctx.template HostAlloc<int>(seq_lod);
    seq_lod_data[0] = 0;
    switch (mask_tensor->dtype()) {
      case DataType::FLOAT32:
        FillSeqLod(batch_size,
                   max_seq_len_value,
                   mask_tensor->data<float>(),
                   seq_lod_data);
        break;
      case DataType::INT64:
        FillSeqLod(batch_size,
                   max_seq_len_value,
                   mask_tensor->data<int64_t>(),
                   seq_lod_data);
        break;
      default:
        PADDLE_THROW(
            phi::errors::Unimplemented("Only support mask data type is int64 "
                                       "or float, not support %s now.",
                                       DataTypeToString(mask_tensor->dtype())));
        break;
    }
    out->Resize({batch_size, seq_lod_data[batch_size], emb_dim});
  }

  int ids_len = id_dims[0] * id_dims[1];
  std::vector<xpu::VectorParam<TID>> arg_ids;
  for (int i = 0; i < emb_layer_num; i++) {
    arg_ids.push_back(
        xpu::VectorParam<TID>{ids[i]->data<TID>(), ids_len, nullptr});
  }

  int r = xpu::multi_embedding_fusion<XPUType, XPUType, TID>(
      ctx.x_context(),
      arg_tables,
      reinterpret_cast<XPUType*>(ctx.template Alloc<TT>(out)),
      arg_ids,
      table_lens,
      emb_dim,
      std::vector<float>(table_lens.size(), 1.0f),
      std::vector<TID>(table_lens.size(), padding_idx));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_with_eltwise_add_xpu");
}
}  // namespace

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
  switch (ids[0]->dtype()) {
    case DataType::INT32:
      MultiEmbeddingKernel<T, int, Context>(
          ctx, ids, tables, mask, padding_idx, out, seq_lod, max_seq_len);
      break;
    case DataType::INT64:
      MultiEmbeddingKernel<T, int64_t, Context>(
          ctx, ids, tables, mask, padding_idx, out, seq_lod, max_seq_len);
      break;
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support ids data type is int64 or int32, not support %s now.",
          DataTypeToString(ids[0]->dtype())));
      break;
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
