/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

template <typename T>
struct EmbeddingVSumFunctor {
  void operator()(const framework::ExecutionContext &context,
                  const LoDTensor *table_t, const LoDTensor *ids_t,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t table_height = table_t->dims()[0];
    int64_t table_width = table_t->dims()[1];
    int64_t out_width = output_t->dims()[1];
    const int64_t *ids = ids_t->data<int64_t>();
    auto ids_lod = ids_t->lod()[0];
    int64_t idx_width = ids_t->numel() / ids_lod.back();
    auto *output = output_t->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_LE(table_width * idx_width, out_width);
    PADDLE_ENFORCE_GT(ids_lod.size(), 1UL);

    jit::emb_seq_pool_attr_t attr(table_height, table_width, 0, idx_width,
                                  out_width, jit::SeqPoolType::kSum);
    for (size_t i = 0; i != ids_lod.size() - 1; ++i) {
      attr.index_height = ids_lod[i + 1] - ids_lod[i];
      auto emb_seqpool = jit::Get<jit::kEmbSeqPool, jit::EmbSeqPoolTuples<T>,
                                  platform::CPUPlace>(attr);
      emb_seqpool(table, ids + ids_lod[i] * idx_width, output + i * out_width,
                  &attr);
    }
  }
};

template <typename T>
class FusedEmbeddingSeqPoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const LoDTensor *ids_t = context.Input<LoDTensor>("Ids");  // int tensor
    LoDTensor *output_t = context.Output<LoDTensor>("Out");    // float tensor
    const LoDTensor *table_var = context.Input<LoDTensor>("W");
    const std::string &combiner_type = context.Attr<std::string>("combiner");

    if (combiner_type == "sum") {
      EmbeddingVSumFunctor<T> functor;
      functor(context, table_var, ids_t, output_t);
    }
  }
};

template <typename T>
class FusedEmbeddingSeqPoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    DDim table_dim;
    if (table_var->IsType<LoDTensor>()) {
      table_dim = context.Input<LoDTensor>("W")->dims();
    } else if (table_var->IsType<SelectedRows>()) {
      auto *table_t = context.Input<SelectedRows>("W");
      table_dim = table_t->value().dims();
    } else {
      PADDLE_THROW(
          "The parameter W of a LookupTable "
          "must be either LoDTensor or SelectedRows");
    }

    bool is_sparse = context.Attr<bool>("is_sparse");
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<SelectedRows>(framework::GradVarName("W"));

      auto *ids_data = ids->data<int64_t>();
      int64_t ids_num = ids->numel();
      auto lod = ids->lod()[0];
      int64_t row_width = d_output->dims()[1];

      framework::Vector<int64_t> *new_rows = d_table->mutable_rows();
      new_rows->resize(ids_num);
      std::memcpy(&(*new_rows)[0], ids_data, ids_num * sizeof(int64_t));

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table_dim[1]});
      T *d_table_data = d_table_value->mutable_data<T>(context.GetPlace());
      const T *d_output_data = d_output->data<T>();

      auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
      for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
        int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
        int64_t in_offset = lod[i] * row_width;
        const T *out_pos = d_output_data + i * row_width;
        T *in_pos = d_table_data + in_offset;
        for (int r = 0; r != h; ++r) {
          blas.VCOPY(row_width, out_pos, in_pos + r * row_width);
        }
      }
    } else {
      LOG(ERROR) << "Dense is not supported in fused_embedding_seq_pool_op now";
    }
  }
};

}  // namespace operators
}  // namespace paddle
