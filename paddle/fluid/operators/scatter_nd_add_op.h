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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/scatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ScatterNdAddOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));
    auto *X = ctx.Input<Tensor>("X");
    auto *Ids = ctx.Input<Tensor>("Index");
    auto *Updates = ctx.Input<Tensor>("Updates");
    auto *Out = ctx.Output<Tensor>("Out");

    // In place output: Out = X
    framework::TensorCopySync(*X, ctx.GetPlace(), Out);
    const auto &index_type = framework::TransToProtoVarType(Ids->dtype());
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Index holds the wrong type, it holds [%s], but "
                          "desires to be [%s] or [%s].",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    if (index_type == framework::proto::VarType::INT32) {
      ScatterNdAdd<T, int32_t>(ctx, *Updates, *Ids, Out);
    } else {
      ScatterNdAdd<T, int64_t>(ctx, *Updates, *Ids, Out);
    }
  }
};

template <typename T>
class ScatterNdAddGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dUpdates = ctx.Output<Tensor>(framework::GradVarName("Updates"));
    auto *Ids = ctx.Input<Tensor>("Index");
    auto *dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));

    if (dX) {
      framework::TensorCopy(*dOut, ctx.GetPlace(), dX);
    }
    if (dUpdates) {
      dUpdates->mutable_data<T>(ctx.GetPlace());
      // Gradient by Gather: dUpdates = dO[Ids]
      const auto &index_type = framework::TransToProtoVarType(Ids->dtype());
      if (index_type == framework::proto::VarType::INT32) {
        CPUGatherNd<T, int32_t>(ctx.device_context(), *dOut, *Ids, dUpdates);
      } else {
        CPUGatherNd<T, int64_t>(ctx.device_context(), *dOut, *Ids, dUpdates);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
