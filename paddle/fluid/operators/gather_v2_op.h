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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class GatherV2OpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* index = ctx.Input<Tensor>("Index");
    auto* axis = ctx.Input<Tensor>("Axis");
    auto* out = ctx.Output<Tensor>("Y");
    auto* axis_data = axis->data<int32_t>();
    auto* index_data = index->data<int32_t>();
    auto* input_data = input->data<T>();

    int axis_size = axis->numel();
    int index_size = index->numel();
    int input_size = input->numel();
    auto input_dim = input->dims();
    if (input->numel() == 0) return;
    PADDLE_ENFORCE_EQ(axis_size, 1,
                      platform::errors::InvalidArgument(
                          "Axis size should be 1, but received %d", axis_size));
    int axis_index = axis;
    auto index_dim_size = input_dim[axis];
    PADDLE_ENFORCE_LE(
        index_size, index_dim_size,
        platform::errors::InvalidArgument(
            "The size that index should be less equal than the dim size of "
            "input,"
            "but received index size:%d, the dim size of input %d.",
            axis_size, index_dim_size));

    int inner_dim_size = 1;
    int outer_dim_size = 1;
    std::vector<int> out_dim_vec = {input_dim_size};

    for (int i = 0; i < axis_index; i++) {
      inner_dim_size *= input_dim[i];
    }
    for (int i = axis_index + 1; i < input_dim.size(); i++) {
      outer_dim_size *= input_dim[i];
      out_dim_vec.push_back(input_dim[i]);
    }
    auto out_dim = framework::make_ddim(out_dim_vec);

    out->Resize(out_dim);
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < inner_dim_size; i++) {
      for (int j = 0; j < outer_dim_size) {
      }
    }
  }
};

template <typename T>
class GatherV2GradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    /*
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto *index = ctx.Input<Tensor>("Index");
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));

    dX->mutable_data<T>(ctx.GetPlace());
    auto dxt = framework::EigenVector<T>::Flatten(*dX);
    auto &place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    dxt.device(place) = dxt.constant(static_cast<T>(0));
    if (dO->numel() == 0) return;
    bool overwrite = ctx.Attr<bool>("overwrite");

    const auto &index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Index holds the wrong type, it holds [%s],"
                          "but desires to be [%s] or [%s].",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));
    if (index_type == framework::proto::VarType::INT32) {
      if (overwrite) {
        ScatterAssign<T, int32_t>(ctx.device_context(), *dO, *index, dX);
      } else {
        ScatterAssignAdd<T, int32_t>(ctx, *dO, *index, dX);
      }
    } else if (index_type == framework::proto::VarType::INT64) {
      if (overwrite) {
        ScatterAssign<T, int64_t>(ctx.device_context(), *dO, *index, dX);
      } else {
        ScatterAssignAdd<T, int64_t>(ctx, *dO, *index, dX);
      }
    }

    */
  }
};

}  // namespace operators
}  // namespace paddle
