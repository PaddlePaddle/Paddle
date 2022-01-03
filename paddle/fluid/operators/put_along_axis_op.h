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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather_scatter_kernel.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class PutAlongAxisOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "PutAlongAxisOpKernel only runs on CPU."));

    auto input = ctx.Input<Tensor>("Input");
    auto axis = ctx.Attr<int>("Axis");
    auto value = ctx.Input<Tensor>("Value");
    auto index = ctx.Input<Tensor>("Index");
    auto reduce_op = ctx.Attr<std::string>("Reduce");
    auto result = ctx.Output<Tensor>("Result");

    framework::TensorCopy(*input, ctx.GetPlace(), result);
    const platform::DeviceContext &device_ctx = ctx.device_context();
    const auto &index_type = index->type();
    if (reduce_op == "add") {
      if (index_type == framework::proto::VarType::INT32) {
        cpu_scatter_add_kernel<T, int32_t>(*result, axis, *index, *value,
                                           device_ctx);
      } else if (index_type == framework::proto::VarType::INT64) {
        cpu_scatter_add_kernel<T, int64_t>(*result, axis, *index, *value,
                                           device_ctx);
      }
    } else if (reduce_op == "multiply" || reduce_op == "mul") {
      if (index_type == framework::proto::VarType::INT32) {
        cpu_scatter_mul_kernel<T, int32_t>(*result, axis, *index, *value,
                                           device_ctx);
      } else if (index_type == framework::proto::VarType::INT64) {
        cpu_scatter_mul_kernel<T, int64_t>(*result, axis, *index, *value,
                                           device_ctx);
      }
    } else if (reduce_op == "assign") {
      if (index_type == framework::proto::VarType::INT32) {
        cpu_scatter_assign_kernel<T, int32_t>(*result, axis, *index, *value,
                                              device_ctx);
      } else if (index_type == framework::proto::VarType::INT64) {
        cpu_scatter_assign_kernel<T, int64_t>(*result, axis, *index, *value,
                                              device_ctx);
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "can not support reduce_op: '%s' for scatter kernel, only "
          "support reduce op: 'add‘, 'assign', 'mul' and 'multiply', the "
          "defalut reduce "
          "op is 'assign' ",
          reduce_op));
      return;
    }
  }
};

template <typename T>
class PutAlongAxisGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "PutAlongAxisGradOpKernel only runs on CPU."));

    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto value_grad = ctx.Output<Tensor>(framework::GradVarName("Value"));
    auto index = ctx.Input<Tensor>("Index");
    auto result_grad = ctx.Input<Tensor>(framework::GradVarName("Result"));
    auto axis = ctx.Attr<int>("Axis");
    const auto &index_type = index->type();

    if (input_grad) {
      framework::TensorCopy(*result_grad, ctx.GetPlace(), input_grad);
      if (index_type == framework::proto::VarType::INT32) {
        cpu_scatter_input_grad_kernel<T, int32_t>(
            // Here passing an unused argument *result_grad, because it's
            // convenient to instantiate a bunch of template function with the
            // same arguments list.
            *result_grad, axis, *index, *input_grad, ctx.device_context());
      } else {
        cpu_scatter_input_grad_kernel<T, int64_t>(
            *result_grad, axis, *index, *input_grad, ctx.device_context());
      }
    }

    if (value_grad) {
      value_grad->Resize(index->dims());
      value_grad->mutable_data<T>(ctx.GetPlace());
      if (index_type == framework::proto::VarType::INT32) {
        cpu_gather_kernel<T, int32_t>(*result_grad, axis, *index, *value_grad,
                                      ctx.device_context());
      } else if (index_type == framework::proto::VarType::INT64) {
        cpu_gather_kernel<T, int64_t>(*result_grad, axis, *index, *value_grad,
                                      ctx.device_context());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
