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

template <typename T, typename IndexT = int>
void IndexSampleInner(const framework::ExecutionContext &context,
                      const LoDTensor &input, const LoDTensor &index,
                      LoDTensor *output) {
  auto input_dims = input.dims();
  auto index_dims = index.dims();

  int batch_size = input_dims[0];
  auto value_length = input_dims[1];
  auto index_length = index_dims[1];
  int index_ids_num = index.numel();
  auto *input_data = input.data<T>();
  auto *index_data = index.data<IndexT>();

  std::vector<T> res{};
  for (int i = 0; i < index_ids_num; i++) {
    int b = floor(i / index_length);
    PADDLE_ENFORCE_GE(
        index_data[i], 0,
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_sample) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            value_length, index_data[i]));
    PADDLE_ENFORCE_LT(
        index_data[i], value_length,
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_sample) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.")),
        value_length, index_data[i]);

    int v_i = b * value_length + static_cast<int>(index_data[i]);
    T v = input_data[v_i];
    VLOG(4) << "Index Sample: batch = " << b << " index = " << v_i
            << " value = " << v;
    res.push_back(v);
  }

  auto ddim = framework::make_ddim({batch_size, index_length});
  output->Resize(ddim);
  T *out_data = output->mutable_data<T>(context.GetPlace());

  memcpy(out_data, &res[0], sizeof(T) * index_ids_num);
}

template <typename DeviceContext, typename T>
class IndexSampleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("X");
    auto *index_var = ctx.InputVar("Index");

    auto &input_tensor = input_var->Get<LoDTensor>();
    auto &index_tensor = index_var->Get<LoDTensor>();

    auto *out_var = ctx.OutputVar("Out");
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();

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
    if (index_type == framework::proto::VarType::INT32) {
      IndexSampleInner<T, int>(ctx, input_tensor, index_tensor, out_tensor);
    } else if (index_type == framework::proto::VarType::INT64) {
      IndexSampleInner<T, int64_t>(ctx, input_tensor, index_tensor, out_tensor);
    }
  }
};

template <typename T, typename IndexT = int>
void IndexSampleGradInner(const framework::ExecutionContext &context,
                          const LoDTensor &out_grad, const LoDTensor &index,
                          LoDTensor *x_grad) {
  auto index_dims = index.dims();
  // auto out_grad_dims = out_grad.dims();
  auto x_grad_dims = x_grad->dims();

  int batch_size = x_grad_dims[0];
  auto value_length = x_grad_dims[1];
  auto index_length = index_dims[1];
  int index_ids_num = index.numel();

  T *x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
  auto *out_grad_data = out_grad.data<T>();
  auto *index_data = index.data<IndexT>();

  memset(x_grad_data, 0, batch_size * value_length * sizeof(T));

  for (int i = 0; i < index_ids_num; i++) {
    int b = floor(i / index_length);
    PADDLE_ENFORCE_GE(
        index_data[i], 0,
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_sample_grad) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            value_length, index_data[i]));
    PADDLE_ENFORCE_LT(
        index_data[i], value_length,
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_sample_grad) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            value_length, index_data[i]));
    int v_i = b * value_length + static_cast<int>(index_data[i]);
    x_grad_data[v_i] += out_grad_data[i];
  }
}

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
    if (index_type == framework::proto::VarType::INT32) {
      IndexSampleGradInner<T, int>(context, out_grad_tensor, index_tensor,
                                   x_grad_tensor);
    } else if (index_type == framework::proto::VarType::INT64) {
      IndexSampleGradInner<T, int64_t>(context, out_grad_tensor, index_tensor,
                                       x_grad_tensor);
    }
  }
};

}  // namespace operators
}  // namespace paddle
