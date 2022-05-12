/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

inline void CheckTableValid() {}

template <typename TIds, typename TData>
void GetIdsEmbedding(const TIds* ids, size_t ids_len, int64_t start_idx,
                     const TData* table, int64_t height, int64_t width,
                     TData* out) {
  for (size_t i = 0; i < ids_len; i++) {
    TIds id = ids[i];
    int64_t local = id - start_idx;

    if (local >= 0 && local < height) {
      // for (int64_t w = 0; w < width; w++) {
      //   out[i * width + w] = table[local * width + w];
      // }

      memcpy(out + i * width, table + local * width, width * sizeof(TData));
    } else {
      memset(out + i * width, 0, width * sizeof(TData));
    }
  }
}

template <typename T>
class CEmbeddingOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* table_t = ctx.Input<LoDTensor>("W");
    auto* ids_t = ctx.Input<LoDTensor>("Ids");
    auto* output_t = ctx.Output<LoDTensor>("Out");
    const int64_t start_idx = ctx.Attr<int64_t>("start_index");

    VLOG(10) << "table_dims:" << table_t->dims();

    const T* table_data = table_t->data<T>();
    T* output_data = output_t->mutable_data<T>(ctx.GetPlace());

    const int64_t height = table_t->dims()[0];
    const int64_t width = table_t->dims()[1];

    const auto& index_type = framework::TransToProtoVarType(ids_t->dtype());
    if (index_type == framework::proto::VarType::INT32) {
      GetIdsEmbedding(ids_t->data<int32_t>(), ids_t->numel(), start_idx,
                      table_data, height, width, output_data);
    } else if (index_type == framework::proto::VarType::INT64) {
      GetIdsEmbedding(ids_t->data<int64_t>(), ids_t->numel(), start_idx,
                      table_data, height, width, output_data);
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "CPU c_embedding ids only support int32 or int64."));
    }
  }
};

template <typename TIds, typename TData>
void UpdateEmbedding(const TIds* ids, size_t ids_len, int64_t start_idx,
                     TData* table, int64_t height, int64_t width,
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

template <typename T>
class CEmbeddingGradOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const int64_t start_idx = context.Attr<int64_t>("start_index");
    auto ids_t = context.Input<LoDTensor>("Ids");
    auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto table_t = context.Input<LoDTensor>("W");
    auto table_grad_t = context.Output<LoDTensor>(framework::GradVarName("W"));

    T* table_grad_data =
        table_grad_t->mutable_data<T>(table_t->dims(), context.GetPlace());

    size_t table_t_mem_size =
        table_t->numel() * framework::DataTypeSize(table_grad_t->dtype());
    size_t table_grad_t_mem_size =
        table_grad_t->numel() *
        framework::SizeOfType(
            framework::TransToProtoVarType(table_grad_t->dtype()));

    VLOG(10) << "table_dims:" << table_t->dims()
             << ", table_t memory_size:" << table_t_mem_size
             << ", table_grad_t memory_size:" << table_grad_t_mem_size
             << ", start_index:" << start_idx;

    memset(table_grad_data, 0, table_grad_t_mem_size);
    const T* d_output_data = d_output_t->data<T>();

    const int64_t height = table_t->dims()[0];
    const int64_t width = table_t->dims()[1];

    const auto& index_type = framework::TransToProtoVarType(ids_t->dtype());
    if (index_type == framework::proto::VarType::INT32) {
      UpdateEmbedding(ids_t->data<int32_t>(), ids_t->numel(), start_idx,
                      table_grad_data, height, width, d_output_data);
    } else if (index_type == framework::proto::VarType::INT64) {
      UpdateEmbedding(ids_t->data<int64_t>(), ids_t->numel(), start_idx,
                      table_grad_data, height, width, d_output_data);
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "CPU c_embedding ids only support int32 or int64."));
    }
  }
};

}  // namespace operators
}  // namespace paddle
