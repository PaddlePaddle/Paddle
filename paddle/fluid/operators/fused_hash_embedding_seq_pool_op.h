/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

extern "C" {
#include <xxhash.h>
}
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
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
                  const int64_t num_hash, const int64_t mod_by,
                  const LoDTensor *table_t, const LoDTensor *in_t,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t row_width = table_t->dims()[1];

    int64_t last_dim = output_t->dims()[1];

    auto *input = in_t->data<int64_t>();
    auto in_lod = in_t->lod()[0];
    PADDLE_ENFORCE_EQ(in_t->dims().size(), 2);
    int64_t seq_num = in_t->dims()[in_t->dims().size() - 1];

    auto *output = output_t->mutable_data<T>(context.GetPlace());

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int64_t i = 0; i != in_lod.size() - 1; ++i) {
      for (int64_t r = in_lod[i]; r < in_lod[i + 1]; ++r) {
        for (int ihash = 0; ihash != num_hash; ++ihash) {
          int64_t id =
              XXH64(input + r * seq_num, sizeof(int) * seq_num, ihash) % mod_by;
          blas.AXPY(row_width, 1., table + id * row_width,
                    output + i * last_dim + ihash * row_width);
        }
      }
    }
  }
};

template <typename T>
class FusedHashEmbeddingSeqPoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    // check input size and each input's dims
    auto in_vars = context.MultiInputVar("X");
    PADDLE_ENFORCE_GE(
        in_vars.size(), 1,
        "FusedHashEmbeddingSeqPoolOp Input(X) should be at least one");
    int64_t seq_length = in_vars[0]->Get<LoDTensor>().dims()[0];
    for (auto *in_var : in_vars) {
      auto in_dims = in_var->Get<LoDTensor>().dims();
      auto in_lod = in_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
          "The actual input data's size mismatched with LoD information.");
      PADDLE_ENFORCE_EQ(
          in_dims[0], seq_length,
          "The actual input datas should have the same batch size");
      PADDLE_ENFORCE_EQ(
          in_dims[0], seq_length,
          "The actual input datas should have the same batch size");
    }

    LoDTensor *output_t = context.Output<LoDTensor>("Out");

    // memset to .0
    output_t->mutable_data<T>(context.GetPlace());
    math::SetConstant<platform::CPUDeviceContext, T> set_constant_functor;
    set_constant_functor(
        context.template device_context<platform::CPUDeviceContext>(), output_t,
        .0);

    const LoDTensor *table_var = context.Input<LoDTensor>("W");
    const std::string &combiner_type = context.Attr<std::string>("combiner");

    if (combiner_type == "sum") {
      for (auto *in_var : in_vars) {
        EmbeddingVSumFunctor<T> functor;
        functor(context, context.Attr<int64_t>("num_hash"),
                context.Attr<int64_t>("mod_by"), table_var,
                &(in_var->Get<LoDTensor>()), output_t);
      }
    } else {
      PADDLE_THROW("The Combiner Type must be sum now.");
    }
  }
};

template <typename T>
class FusedHashEmbeddingSeqPoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    int64_t num_hash = context.Attr<int64_t>("num_hash");
    int64_t mod_by = context.Attr<int64_t>("mod_by");

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
      auto in_vars = context.MultiInputVar("X");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<SelectedRows>(framework::GradVarName("W"));

      auto lod = in_vars[0]->Get<LoDTensor>().lod()[0];
      int64_t ids_num = lod.back() * num_hash * in_vars.size();

      std::vector<int64_t> new_rows;
      new_rows.reserve(ids_num);
      for (auto in_var : in_vars) {
        const LoDTensor &in_tensor = in_var->Get<LoDTensor>();
        const int64_t *input = in_tensor.data<int64_t>();
        int64_t seq_num = in_tensor.dims()[in_tensor.dims().size() - 1];
        for (int i = 0; i != lod.back(); ++i) {
          for (int ihash = 0; ihash != num_hash; ++ihash) {
            new_rows.emplace_back(
                XXH64(input + i * seq_num, sizeof(int) * seq_num, ihash) %
                mod_by);
          }
        }
      }
      d_table->set_rows(new_rows);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table_dim[1]});
      T *d_table_data = d_table_value->mutable_data<T>(context.GetPlace());
      const T *d_output_data = d_output->data<T>();
      int64_t row_width = d_output->dims()[1];

      auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
      for (int i = 0; i != in_vars.size(); ++i) {
        T *d_table_data_pos = d_table_data + i * lod.back() * row_width;
        for (int j = 0; j < static_cast<int>(lod.size()) - 1; ++j) {
          int64_t h = static_cast<int64_t>(lod[j + 1] - lod[j]);
          int64_t in_offset = lod[j] * row_width;
          const T *out_pos = d_output_data + j * row_width;
          T *in_pos = d_table_data_pos + in_offset;
          for (int r = 0; r != h; ++r) {
            blas.VCOPY(row_width, out_pos, in_pos + r * row_width);
          }
        }
      }
    } else {
      LOG(ERROR) << "Dense is not supported in fused_embedding_seq_pool_op now";
    }
  }
};

}  // namespace operators
}  // namespace paddle
