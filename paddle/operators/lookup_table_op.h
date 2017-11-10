/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

#include "paddle/framework/eigen.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/selected_rows.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

template <typename T>
class LookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* table_t = context.Input<LoDTensor>("W");      // float tensor
    auto* ids_t = context.Input<LoDTensor>("Ids");      // int tensor
    auto* output_t = context.Output<LoDTensor>("Out");  // float tensor

    int N = table_t->dims()[0];
    int D = table_t->dims()[1];
    auto* ids = ids_t->data<int64_t>();
    auto* table = table_t->data<T>();
    auto* output = output_t->mutable_data<T>(context.GetPlace());
    for (int64_t i = 0; i < ids_t->numel(); ++i) {
      PADDLE_ENFORCE_LT(ids[i], N);
      PADDLE_ENFORCE_GE(ids[i], 0);
      memcpy(output + i * D, table + ids[i] * D, D * sizeof(T));
    }
  }
};

template <typename T>
class LookupTableGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool is_sparse = context.Attr<bool>("is_sparse");
    if (is_sparse) {
      auto* ids = context.Input<LoDTensor>("Ids");
      auto* table = context.Input<LoDTensor>("W");
      auto* d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto* d_table = context.Output<SelectedRows>(framework::GradVarName("W"));

      auto* ids_data = ids->data<int64_t>();
      auto ids_dim = ids->dims();

      framework::Vector<int64_t> new_rows;
      new_rows.reserve(ids_dim[0]);
      for (int64_t i = 0; i < ids_dim[0]; i++) {
        new_rows.push_back(ids_data[i]);
      }
      d_table->set_rows(new_rows);

      auto* d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_dim[0], table->dims()[1]});
      d_table_value->mutable_data<T>(context.GetPlace());

      d_table->set_height(table->dims()[0]);

      auto* d_output_data = d_output->data<T>();
      auto* d_table_data = d_table_value->data<T>();

      PADDLE_ENFORCE_EQ(d_table_value->dims(), d_output->dims());
      memcpy(d_table_data, d_output_data, sizeof(T) * d_output->numel());
    } else {
      auto* ids = context.Input<LoDTensor>("Ids");
      auto* d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto* d_table = context.Output<LoDTensor>(framework::GradVarName("W"));
      auto* table = context.Input<LoDTensor>("W");

      auto* ids_data = ids->data<int64_t>();
      auto ids_dim = ids->dims();

      int N = table->dims()[0];
      int D = d_output->dims()[1];

      auto* d_output_data = d_output->data<T>();
      auto* d_table_data = d_table->mutable_data<T>(context.GetPlace());

      memset(d_table_data, 0, d_table->numel() * sizeof(T));

      for (int64_t i = 0; i < ids->numel(); ++i) {
        PADDLE_ENFORCE_LT(ids_data[i], N);
        PADDLE_ENFORCE_GE(ids_data[i], 0);
        for (int j = 0; j < D; ++j) {
          d_table_data[ids_data[i] * D + j] += d_output_data[i * D + j];
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
