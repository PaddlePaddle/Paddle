/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather_scatter_kernel.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class TakeAlongAxisOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto input = ctx.Input<Tensor>("Input");
    auto axis = ctx.Attr<int>("Axis");
    auto index = ctx.Input<Tensor>("Index");
    auto result = ctx.Output<Tensor>("Result");
    result->Resize(index->dims());
    result->mutable_data<T>(ctx.GetPlace());

    const auto &index_type = index->type();
    if (index_type == framework::proto::VarType::INT32) {
      cpu_gather_kernel<T, int32_t>(*input, axis, *index, *result,
                                    ctx.device_context());
    } else if (index_type == framework::proto::VarType::INT64) {
      cpu_gather_kernel<T, int64_t>(*input, axis, *index, *result,
                                    ctx.device_context());
    }
  }
};

template <typename T>
class TakeAlongAxisGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto index = ctx.Input<Tensor>("Index");
    auto result_grad = ctx.Input<Tensor>(framework::GradVarName("Result"));
    auto axis = ctx.Attr<int>("Axis");
    // We need to know the shape of input matrix to determine the shape of grad
    // matrix of input.
    auto input = ctx.Input<Tensor>("Input");
    input_grad->Resize(input->dims());
    input_grad->mutable_data<T>(ctx.GetPlace());

    // Set to zero tensor.
    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> functor;
    functor(reinterpret_cast<const platform::CPUDeviceContext &>(dev_ctx),
            input_grad, static_cast<T>(0));

    const auto &index_type = index->type();
    if (index_type == framework::proto::VarType::INT32) {
      cpu_scatter_add_kernel<T, int32_t>(
          *input_grad, axis, *index, *result_grad,
          ctx.device_context());  // the gradient of gather is scatter
    } else if (index_type == framework::proto::VarType::INT64) {
      cpu_scatter_add_kernel<T, int64_t>(*input_grad, axis, *index,
                                         *result_grad, ctx.device_context());
    }
  }
};

}  // namespace operators
}  // namespace paddle
