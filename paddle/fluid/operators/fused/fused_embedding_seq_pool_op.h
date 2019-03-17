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
#include "paddle/fluid/operators/math/math_function.h"

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
    auto *output = output_t->mutable_data<T>(context.GetPlace());
    auto& dev_ctx =
            context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(dev_ctx, reinterpret_cast<Tensor *>(output_t), static_cast<T>(0));
    if (ids_t->numel() == 0 || ids_lod.back() == 0) {
      return;  
    }
    int64_t idx_width = ids_t->numel() / ids_lod.back();


    PADDLE_ENFORCE_LE(table_width * idx_width, out_width);
    PADDLE_ENFORCE_GT(ids_lod.size(), 1UL, "The LoD[0] could NOT be empty");

    jit::emb_seq_pool_attr_t attr(table_height, table_width, 0, idx_width,
                                  out_width, jit::SeqPoolType::kSum);
    for (size_t i = 0; i != ids_lod.size() - 1; ++i) {
      attr.index_height = ids_lod[i + 1] - ids_lod[i];
      if (attr.index_height > 0) {
        auto emb_seqpool = jit::Get<jit::kEmbSeqPool, jit::EmbSeqPoolTuples<T>,
                                  platform::CPUPlace>(attr);
        emb_seqpool(table, ids + ids_lod[i] * idx_width, output + i * out_width,
                  &attr);
      }
    }
  }
};

inline int FusedEmbeddingSeqPoolLastDim(const framework::DDim &table_dims,
                                        const framework::DDim &ids_dims) {
  int64_t last_dim = table_dims[1];
  for (int i = 1; i != ids_dims.size(); ++i) {
    last_dim *= ids_dims[i];
  }
  return last_dim;
}

template <typename T>
class FusedEmbeddingSeqPoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const LoDTensor *ids_t = context.Input<LoDTensor>("Ids");  // int tensor
    LoDTensor *output_t = context.Output<LoDTensor>("Out");    // float tensor
    const LoDTensor *table_var = context.Input<LoDTensor>("W");
    const std::string &combiner_type = context.Attr<std::string>("combiner");

    int64_t last_dim =
        FusedEmbeddingSeqPoolLastDim(table_var->dims(), ids_t->dims());
    const auto &ids_lod = ids_t->lod();
    // in run time, the LoD of ids must be 1
    PADDLE_ENFORCE(ids_lod.size(), 1UL,
                   "The LoD level of Input(Ids) must be 1");
    int64_t batch_size = ids_lod[0].size() - 1;
    // in run time, the shape from Ids -> output
    // should be [seq_length, 1] -> [batch_size, last_dim]
    output_t->Resize({batch_size, last_dim});

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
      int64_t out_width = d_output->dims()[1];

      framework::Vector<int64_t> *new_rows = d_table->mutable_rows();
      new_rows->resize(ids_num);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table_dim[1]});
      T *d_table_data = d_table_value->mutable_data<T>(context.GetPlace());
      const T *d_output_data = d_output->data<T>();

      if (ids_num == 0) {
        return;
      }

      std::memcpy(&(*new_rows)[0], ids_data, ids_num * sizeof(int64_t));
      auto vbroadcast = jit::Get<jit::kVBroadcast, jit::VBroadcastTuples<T>,
                                 platform::CPUPlace>(out_width);
      for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
        int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
        if (h > 0) {
          const T *src = d_output_data + i * out_width;
          T *dst = d_table_data + lod[i] * out_width;
          vbroadcast(src, dst, h, out_width);
        }
      }
    } else {
      LOG(ERROR) << "Dense is not supported in fused_embedding_seq_pool_op now";
    }
  }
};

}  // namespace operators
}  // namespace paddle
