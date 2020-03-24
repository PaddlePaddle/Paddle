/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <gflags/gflags.h>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class IndexSampleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("X");
    auto *index_var = ctx.InputVar("Index");

    auto &input_tensor = input_var->Get<LoDTensor>();
    auto &index_tensor = index_var->Get<LoDTensor>();
    auto input_dims = input_tensor.dims();
    auto index_dims = index_tensor.dims();

    int batch_size = input_dims[0];
    auto value_length = input_dims[1];
    auto index_length = index_dims[1];
    int index_ids_num = index_tensor.numel();

    auto *input_data = input_tensor.data<T>();

    const auto &index_type = index_tensor.type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto *index_data = index_type == framework::proto::VarType::INT32
                           ? index_tensor.data<int>()
                           : index_tensor.data<int64_t>();

    std::vector<T> res{};
    for (int i = 0; i < index_ids_num; i++) {
      int b = floor(i / index_length);
      PADDLE_ENFORCE_LT(
          -1, index_data[i],
          "Variable value (index) of OP(index_sample) "
          "expected >= 0 and < %ld, but got %ld. Please check input "
          "value.",
          value_length, index_data[i]);
      PADDLE_ENFORCE_LT(
          index_data[i], value_length,
          "Variable value (index) of OP(index_sample) "
          "expected >= 0 and < %ld, but got %ld. Please check input "
          "value.",
          value_length, index_data[i]);

      int v_i = b * value_length + static_cast<int>(index_data[i]);
      T v = input_data[v_i];
      VLOG(4) << "Index Sample: batch = " << b << " index = " << v_i
              << " value = " << v;
      res.push_back(v);
    }

    auto *out_var = ctx.OutputVar("Out");
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();
    auto ddim = framework::make_ddim({batch_size, index_length});
    out_tensor->Resize(ddim);
    auto *out_data = out_tensor->mutable_data<T>(ctx.GetPlace());

    memcpy(out_data, &res[0], sizeof(T) * index_ids_num);
  }
};

template <typename DeviceContext, typename T>
class IndexSampleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *index_var = context.InputVar("Index");
    auto *x_grad_var = context.OutputVar(framework::GradVarName("X"));
    auto *out_grad_var = context.InputVar(framework::GradVarName("Out"));

    auto &index_tensor = index_var->Get<LoDTensor>();
    auto &out_grad_tensor = out_grad_var->Get<LoDTensor>();
    auto *x_grad_tensor = x_grad_var->GetMutable<framework::LoDTensor>();

    auto index_dims = index_tensor.dims();
    auto out_grad_dims = out_grad_tensor.dims();
    auto x_grad_dims = x_grad_tensor.dims();

    int batch_size = x_grad_dims[0];
    auto value_length = x_grad_dims[1];
    auto index_length = index_dims[1];
    int index_ids_num = index_tensor.numel();

    auto *x_grad_data = x_grad_tensor.data<T>();
    auto *out_grad_data = out_grad_tensor.data<T>();

    const auto &index_type = index_tensor.type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));
    auto *index_data = index_type == framework::proto::VarType::INT32
                           ? index_tensor.data<int>()
                           : index_tensor.data<int64_t>();

    memset(x_grad_data, 0, batch_size * value_length * sizeof(T));

    for (int i = 0; i < index_ids_num, i++) {
      int b = floor(i / index_length);
      PADDLE_ENFORCE_LT(
          -1, index_data[i],
          "Variable value (index) of OP(index_sample_grad) "
          "expected >= 0 and < %ld, but got %ld. Please check input "
          "value.",
          value_length, index_data[i]);
      PADDLE_ENFORCE_LT(
          index_data[i], value_length,
          "Variable value (index) of OP(index_sample_grad) "
          "expected >= 0 and < %ld, but got %ld. Please check input "
          "value.",
          value_length, index_data[i]);
      int v_i = b * value_length + static_cast<int>(index_data[i]);
      x_grad_data[v_i] += out_grad_data[i];
    }
  }
};

}  // namespace operators
}  // namespace paddle
