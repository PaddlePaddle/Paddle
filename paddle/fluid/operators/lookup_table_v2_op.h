/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/math/blas.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/fluid/operators/distributed/parameter_prefetch.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T>
class LookupTableV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ids_t = context.Input<LoDTensor>("Ids");      // int tensor
    auto *output_t = context.Output<LoDTensor>("Out");  // float tensor
    auto *table_var = context.InputVar("W");

    auto id_name = context.Inputs("Ids").front();
    auto embedding_name = context.Inputs("W").front();
    auto out_name = context.Outputs("Out").front();

    // for remote prefetch
    auto epmap = context.Attr<std::vector<std::string>>("epmap");
    auto remote_prefetch = context.Attr<bool>("remote_prefetch");
    auto height_sections =
        context.Attr<std::vector<int64_t>>("height_sections");
    auto table_names = context.Attr<std::vector<std::string>>("table_names");

    if (remote_prefetch && !epmap.empty()) {
// if epmap is not empty, then the parameter will be fetched from remote
// parameter server

#ifdef PADDLE_WITH_DISTRIBUTE
      operators::distributed::prefetch(id_name, out_name, embedding_name, false,
                                       table_names, epmap, height_sections,
                                       context, context.scope());
#else
      PADDLE_THROW(
          "paddle is not compiled with distribute support, can not do "
          "parameter prefetch!");
#endif
    } else {
      int64_t padding_idx = context.Attr<int64_t>("padding_idx");
      int64_t *ids = const_cast<int64_t *>(ids_t->data<int64_t>());
      int64_t ids_numel = ids_t->numel();

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
            PADDLE_ENFORCE_LT(
                ids[i], row_number,
                "Variable value (input) of OP(fluid.layers.embedding) "
                "expected >= 0 and < %ld, but got %ld. Please check input "
                "value.",
                row_number, ids[i]);
            PADDLE_ENFORCE_GE(
                ids[i], 0,
                "Variable value (input) of OP(fluid.layers.embedding) "
                "expected >= 0 and < %ld, but got %ld. Please check input "
                "value.",
                row_number, ids[i]);
            memcpy(output + i * row_width, table + ids[i] * row_width,
                   row_width * sizeof(T));
          }
        }
      } else if (table_var->IsType<SelectedRows>()) {
        const auto &table_t = table_var->Get<SelectedRows>();
        int64_t row_width = table_t.value().dims()[1];
        const auto *table = table_t.value().data<T>();
        auto *output = output_t->mutable_data<T>(context.GetPlace());

        auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
        for (int64_t i = 0; i < ids_numel; ++i) {
          if (padding_idx != kNoPadding && ids[i] == padding_idx) {
            memset(output + i * row_width, 0, row_width * sizeof(T));
          } else {
            PADDLE_ENFORCE_GE(ids[i], 0);
            auto id_index = table_t.Index(ids[i]);
            PADDLE_ENFORCE_GE(id_index, 0, "the input key should be exists.");
            blas.VCOPY(row_width, table + id_index * row_width,
                       output + i * row_width);
          }
        }
      }
    }
  }
};

template <typename T>
class LookupTableV2GradKernel : public framework::OpKernel<T> {
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
          "The parameter W of a LookupTableV2 "
          "must be either LoDTensor or SelectedRows");
    }

    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    bool is_sparse = context.Attr<bool>("is_sparse");
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<SelectedRows>(framework::GradVarName("W"));

      auto *ids_data = ids->data<int64_t>();
      int64_t ids_num = ids->numel();

      std::vector<int64_t> new_rows;
      new_rows.resize(ids_num);
      std::memcpy(&new_rows[0], ids_data, ids_num * sizeof(int64_t));
      d_table->set_rows(new_rows);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table_dim[1]});

      d_table_value->mutable_data<T>(context.GetPlace());

      d_table->set_height(table_dim[0]);

      auto *d_output_data = d_output->data<T>();
      auto *d_table_data = d_table_value->data<T>();

      auto d_output_dims = d_output->dims();
      PADDLE_ENFORCE_EQ(
          d_table_value->dims(),
          framework::flatten_to_2d(d_output_dims, d_output_dims.size() - 1));
      memcpy(d_table_data, d_output_data, sizeof(T) * d_output->numel());

    } else {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<LoDTensor>(framework::GradVarName("W"));

      auto *ids_data = ids->data<int64_t>();

      int64_t N = table_dim[0];
      int64_t D = table_dim[1];

      auto *d_output_data = d_output->data<T>();
      auto *d_table_data = d_table->mutable_data<T>(context.GetPlace());

      memset(d_table_data, 0, d_table->numel() * sizeof(T));

      for (int64_t i = 0; i < ids->numel(); ++i) {
        if (padding_idx != kNoPadding && ids_data[i] == padding_idx) {
          // the gradient of padding_idx should be 0, already done by memset, so
          // do nothing.
        } else {
          PADDLE_ENFORCE_LT(
              ids_data[i], N,
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input value.",
              N, ids_data[i]);
          PADDLE_ENFORCE_GE(
              ids_data[i], 0,
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input value.",
              N, ids_data[i]);
          for (int j = 0; j < D; ++j) {
            d_table_data[ids_data[i] * D + j] += d_output_data[i * D + j];
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
