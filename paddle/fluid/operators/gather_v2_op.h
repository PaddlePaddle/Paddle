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
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename U, typename V>
void GatherV2Function(const Tensor* input, const Tensor* index,
                      const Tensor* axis, Tensor* out,
                      const paddle::platform::Place& place) {
  auto* axis_data = axis->data<V>();
  auto* index_data = index->data<U>();

  int axis_size = axis->numel();
  int index_size = index->numel();
  int input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();

  if (input->numel() == 0) return;
  PADDLE_ENFORCE_EQ(axis_size, 1,
                    platform::errors::InvalidArgument(
                        "Axis size should be 1, but received %d", axis_size));
  int axis_index = axis_data[0];
  int index_dim_size = input_dim[axis_index];
  PADDLE_ENFORCE_LE(
      index_size, index_dim_size,
      platform::errors::InvalidArgument(
          "The size that index should be less equal than the dim size of "
          "input,"
          "but received index size:%d, the dim size of input %d.",
          axis_size, index_dim_size));

  int inner_dim_size = 1;
  int outer_dim_size = 1;
  std::vector<int> out_dim_vec;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  out_dim_vec.push_back(index_size);
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  auto out_dim = framework::make_ddim(out_dim_vec);

  out->Resize(out_dim);
  auto* out_data = out->mutable_data<T>(place);

  int out_index = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < index_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    (i * input_size / inner_dim_size);
        out_data[out_index] = input_data[index];
        out_index++;
      }
    }
  }
}

template <typename T>
class GatherV2OpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* index = ctx.Input<Tensor>("Index");
    const Tensor* axis = ctx.Input<Tensor>("Axis");
    Tensor* out = ctx.Output<Tensor>("Y");
    const Tensor* input = ctx.Input<Tensor>("X");

    const auto& index_type = index->type();
    const auto& axis_type = axis->type();
    auto place = ctx.GetPlace();
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2Function<T, int32_t, int32_t>(input, index, axis, out, place);
    }
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2Function<T, int32_t, int64_t>(input, index, axis, out, place);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2Function<T, int64_t, int32_t>(input, index, axis, out, place);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2Function<T, int64_t, int64_t>(input, index, axis, out, place);
    }
  }
};

template <typename T, typename U, typename V>
void GatherV2GradFunction(const Tensor* input, const Tensor* index,
                          const Tensor* axis, Tensor* out,
                          const paddle::platform::Place& place) {
  auto* axis_data = axis->data<V>();
  auto* index_data = index->data<U>();

  int axis_size = axis->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();

  if (input->numel() == 0) return;
  PADDLE_ENFORCE_EQ(axis_size, 1,
                    platform::errors::InvalidArgument(
                        "Axis size should be 1, but received %d", axis_size));
  int axis_index = axis_data[0];
  int input_index_dim_size = input_dim[axis_index];

  int inner_dim_size = 1;
  int outer_dim_size = 1;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  auto* out_data = out->mutable_data<T>(place);
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto out_dim = out->dims();
  int out_index_dim_size = out_dim[axis_index];
  operators::math::set_constant(*dev_ctx, out, 0.0);

  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < input_index_dim_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    i * outer_dim_size * out_index_dim_size;
        out_data[index] += input_data[j * outer_dim_size + k];
      }
    }
  }
}

template <typename T>
class GatherV2GradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* index = ctx.Input<Tensor>("Index");
    auto* axis = ctx.Input<Tensor>("Axis");
    auto* out = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* input = ctx.Input<Tensor>(framework::GradVarName("Y"));

    const auto& index_type = index->type();
    const auto& axis_type = axis->type();
    auto place = ctx.GetPlace();
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2GradFunction<T, int32_t, int32_t>(input, index, axis, out, place);
    }
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2GradFunction<T, int32_t, int64_t>(input, index, axis, out, place);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2GradFunction<T, int64_t, int32_t>(input, index, axis, out, place);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2GradFunction<T, int64_t, int64_t>(input, index, axis, out, place);
    }
  }
};

}  // namespace operators
}  // namespace paddle
