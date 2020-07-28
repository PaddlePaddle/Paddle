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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/common_infer_kernel_type_functions.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

// This file almostly contains all the coefficient-wise Operator class and
// OpKernel class

namespace paddle {
namespace operators {
using Tensor = framework::LoDTensor;

class BinaryOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    return BinaryOpBroadcastInferShape(ctx);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return BinaryOpInferKernelType(ctx);
  }
};

template <typename DeviceContext, typename Functor, typename T>
class BinaryOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>(ctx.GetInputNameByIdx(0));
    auto *y = ctx.Input<Tensor>(ctx.GetInputNameByIdx(1));
    auto *out = ctx.Output<Tensor>(ctx.GetOutputNameByIdx(0));
    out->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<Functor, DeviceContext, T>(ctx, x, y, axis, Functor(),
                                                    out);
  }
};

}  // namespace operators
}  // namespace paddle
