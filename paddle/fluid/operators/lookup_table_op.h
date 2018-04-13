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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T>
class LookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    auto *ids_var = context.InputVar("Ids");
    Tensor *output_t = context.Output<Tensor>("Out");
    int64_t padding_idx = context.Attr<int64_t>("padding_idx");

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

    int64_t *ids;
    int64_t ids_numel;

    // The type of Ids(Input) is SelectedRows or LoDTensor, when Ids's type
    // is LoDTensor, this tensor contains the ids to be looked up in W;
    // when Ids's type is SelectedRows, the rows of Ids contains the
    // ids to be looked up in W.
    if (ids_var->IsType<LoDTensor>()) {
      auto *ids_t = context.Input<LoDTensor>("Ids");
      ids = const_cast<int64_t *>(ids_t->data<int64_t>());
      ids_numel = ids_t->numel();
    } else if (ids_var->IsType<SelectedRows>()) {
      auto *ids_t = context.Input<SelectedRows>("Ids");
      ids = const_cast<int64_t *>(ids_t->rows().data());
      ids_numel = ids_t->rows().size();
      output_t->Resize({ids_numel, table_dim[1]});
    } else {
      PADDLE_THROW("Unsupported Variable Type of Ids");
    }

    if (table_var->IsType<LoDTensor>()) {
      auto *table_t = context.Input<LoDTensor>("W");
      int64_t row_number = table_t->dims()[0];
      int64_t row_width = table_t->dims()[1];

      auto *table = table_t->data<T>();
      auto *output = output_t->mutable_data<T>(context.GetPlace());

      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_LT(ids[i], row_number);
          PADDLE_ENFORCE_GE(ids[i], 0);
          memcpy(output + i * row_width, table + ids[i] * row_width,
                 row_width * sizeof(T));
        }
      }
    } else if (table_var->IsType<SelectedRows>()) {
      const auto &table_t = table_var->Get<SelectedRows>();
      int64_t row_width = table_t.value().dims()[1];
      const auto *table = table_t.value().data<T>();
      auto *output = output_t->mutable_data<T>(context.GetPlace());

      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_GE(ids[i], 0);
          auto id_index = table_t.Index(ids[i]);
          PADDLE_ENFORCE_GE(id_index, 0, "the input key should be exists.");
          memcpy(output + i * row_width, table + id_index * row_width,
                 row_width * sizeof(T));
        }
      }
    }
  }
};

template <typename T>
class LookupTableGradKernel : public framework::OpKernel<T> {
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
      auto ids_dim = ids->dims();

      framework::Vector<int64_t> new_rows;
      new_rows.reserve(ids_dim[0]);
      for (int64_t i = 0; i < ids_dim[0]; i++) {
        new_rows.push_back(ids_data[i]);
      }
      d_table->set_rows(new_rows);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_dim[0], table_dim[1]});
      d_table_value->mutable_data<T>(context.GetPlace());

      d_table->set_height(table_dim[0]);

      auto *d_output_data = d_output->data<T>();
      auto *d_table_data = d_table_value->data<T>();

      PADDLE_ENFORCE_EQ(d_table_value->dims(), d_output->dims());
      memcpy(d_table_data, d_output_data, sizeof(T) * d_output->numel());
    } else {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<LoDTensor>(framework::GradVarName("W"));

      auto *ids_data = ids->data<int64_t>();
      auto ids_dim = ids->dims();

      int N = table_dim[0];
      int D = d_output->dims()[1];

      auto *d_output_data = d_output->data<T>();
      auto *d_table_data = d_table->mutable_data<T>(context.GetPlace());

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
