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
                  const LoDTensor *table_t, const LoDTensor *ids_t, const int rand_len,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t row_number = table_t->dims()[0];
    int64_t row_width = rand_len;
    int64_t last_dim = output_t->dims()[1];
    const int64_t *ids = ids_t->data<int64_t>();
    auto ids_lod = ids_t->lod()[0];
    auto *output = output_t->mutable_data<T>(context.GetPlace());
    auto& dev_ctx =
            context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(dev_ctx, reinterpret_cast<Tensor *>(output_t), static_cast<T>(0));
    if (ids_lod.back() == 0 || ids_t->numel() == 0 || ids_t->lod().size() == 0) {
        
        return;
    }
    int64_t ids_count = ids_t->numel() / ids_lod.back();

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int64_t i = 0; i != ids_lod.size() - 1; ++i) {
      size_t begin = ids_lod[i] * ids_count;
      for (int64_t j = 0; j != ids_count; ++j) {
        PADDLE_ENFORCE_LT(ids[begin], row_number);
        PADDLE_ENFORCE_GE(ids[begin], 0, "ids %d", i);
        blas.VCOPY(row_width, table + ids[begin + j],
                   output + i * last_dim + j * row_width);
      }

      for (int64_t r = (ids_lod[i] + 1) * ids_count;
           r < ids_lod[i + 1] * ids_count; ++r) {
        PADDLE_ENFORCE_LT(ids[r], row_number);
        PADDLE_ENFORCE_GE(ids[r], 0, "ids %d", i);
        blas.AXPY(row_width, 1., table + ids[r],
                  output + i * last_dim + (r % ids_count) * row_width);
      }
    }
  }
};

template <typename T>
struct EmbeddingVSelectFunctor {
  void operator()(const framework::ExecutionContext &context,
                  const LoDTensor *table_t, const LoDTensor *ids_t, const int rand_len, const int select,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t row_number = table_t->dims()[0];
    int64_t row_width = rand_len;
    int64_t last_dim = output_t->dims()[1];
    const int64_t *ids = ids_t->data<int64_t>();
    auto ids_lod = ids_t->lod()[0];
    auto *output = output_t->mutable_data<T>(context.GetPlace());
    auto& dev_ctx =
            context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(dev_ctx, reinterpret_cast<Tensor *>(output_t), static_cast<T>(0));
    if (ids_lod.back() == 0 || ids_t->numel() == 0 || ids_t->lod().size() == 0) {
        return;
    }
    int64_t ids_count = ids_t->numel() / ids_lod.back();

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int64_t i = 0; i != ids_lod.size() - 1; ++i) {
      size_t begin = (ids_lod[i] + select) * ids_count;
      for (int64_t j = 0; j != ids_count; ++j) {
          PADDLE_ENFORCE_LT(ids[begin], row_number);
          PADDLE_ENFORCE_GE(ids[begin], 0, "ids %d", i);
          blas.VCOPY(row_width, table + ids[begin + j],
                     output + i * last_dim + j * row_width);
      }
    }
  }
};

inline int RandomEmbeddingSeqPoolLastDim(const framework::DDim &table_dims,
                                        const framework::DDim &ids_dims, const int rand_len) {
  int64_t last_dim = rand_len;
  VLOG(1) << "qxz rand_len " << rand_len;
  for (int i = 1; i != ids_dims.size(); ++i) {
    last_dim *= ids_dims[i];
    VLOG(1) << "qxz ids_dims " << i << " " << ids_dims[i];
  }
  return last_dim;
}

template <typename T>
class RandomEmbeddingSeqPoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const LoDTensor *ids_t = context.Input<LoDTensor>("Ids");  // int tensor
    LoDTensor *output_t = context.Output<LoDTensor>("Out");    // float tensor
    const LoDTensor *table_var = context.Input<LoDTensor>("W");
    const std::string &combiner_type = context.Attr<std::string>("combiner");
    const int rand_len = context.Attr<int>("rand_len");
    const int select_idx = context.Attr<int>("select_idx");

    int64_t last_dim =
        RandomEmbeddingSeqPoolLastDim(table_var->dims(), ids_t->dims(), rand_len);
    const auto &ids_lod = ids_t->lod();
    // in run time, the LoD of ids must be 1
    PADDLE_ENFORCE(ids_lod.size(), 1UL,
                   "The LoD level of Input(Ids) must be 1");
    PADDLE_ENFORCE(table_var->dims()[1], 1UL,
                   "The last dim of Parameter W must be 1");
    int64_t batch_size = ids_lod[0].size() - 1;
    // in run time, the shape from Ids -> output
    // should be [seq_length, 1] -> [batch_size, last_dim]
    VLOG(1) << "qxz output_size " << batch_size * last_dim << ", " << last_dim;
    if (batch_size != 1) {
       VLOG(1) << "qxz output_size ";
    }
    output_t->Resize({batch_size, last_dim});

    if (combiner_type == "sum") {
      EmbeddingVSumFunctor<T> functor;
      functor(context, table_var, ids_t, rand_len, output_t);
    } else if (combiner_type == "select") {
      EmbeddingVSelectFunctor<T> functor;
      functor(context, table_var, ids_t, rand_len, select_idx, output_t);
    }

  }
};

template <typename T>
class RandomEmbeddingSeqPoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    DDim table_dim;
    if (table_var->IsType<LoDTensor>()) {
      table_dim = context.Input<LoDTensor>("W")->dims();
    } else {
      PADDLE_THROW(
          "The parameter W of a LookupTable "
          "must be LoDTensor");
    }
    PADDLE_ENFORCE(table_dim[1], 1UL,
                   "The last dim of Parameter W must be 1");


    bool is_sparse = context.Attr<bool>("is_sparse");
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<SelectedRows>(framework::GradVarName("W"));
      const int rand_len = context.Attr<int>("rand_len");

      auto *ids_data = ids->data<int64_t>();
      int64_t ids_num = ids->numel();
      auto lod = ids->lod()[0];
      int64_t out_width = d_output->dims()[1];
     
      int64_t new_ids_num = ids_num * rand_len;
      framework::Vector<int64_t> *new_rows = d_table->mutable_rows();
      int64_t row_number = table_dim[0];
      new_rows->resize(new_ids_num);
      int64_t new_idx = 0;
      for (int i = 0; i < new_ids_num - 1; ++i) {
         new_idx = ids_data[i / rand_len] + i % rand_len; 
         PADDLE_ENFORCE_LT(new_idx, row_number);
         (*new_rows)[i] = new_idx; 
      }

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({new_ids_num, table_dim[1]});
      T *d_table_data = d_table_value->mutable_data<T>(context.GetPlace());
      const T *d_output_data = d_output->data<T>();

      auto vbroadcast = jit::Get<jit::kVBroadcast, jit::VBroadcastTuples<T>,
                                 platform::CPUPlace>(out_width);
      for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
        int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
        const T *src = d_output_data + i * out_width;
        T *dst = d_table_data + lod[i] * out_width;
        vbroadcast(src, dst, h, out_width);
      }
    } else {
      LOG(ERROR) << "Dense is not supported in random_embedding_seq_pool_op now";
    }
  }
};

}  // namespace operators
}  // namespace paddle
