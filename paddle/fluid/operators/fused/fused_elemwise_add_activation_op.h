/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/fused/fused_elemwise_activation_op.h"
#include "paddle/fluid/operators/math/compound_functors.h"
#include "paddle/fluid/operators/math/functors.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedElemwiseAddActivationKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &in_x = GET_DATA_SAFELY(ctx.Input<framework::Tensor>("X"), "Input",
                                 "X", "FusedElemwiseActivation");
    auto &in_y = GET_DATA_SAFELY(ctx.Input<framework::Tensor>("Y"), "Input",
                                 "Y", "FusedElemwiseActivation");

    PADDLE_ENFORCE_EQ(ctx.HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "The output(Out) should not be empty"));
    auto output = ctx.Output<framework::Tensor>("Out");

    std::vector<framework::Tensor *> outputs;
    outputs.emplace_back(output);

    if (ctx.Attr<bool>("save_intermediate_out")) {
      PADDLE_ENFORCE_EQ(ctx.HasOutput("IntermediateOut"), true,
                        platform::errors::InvalidArgument(
                            "The save_intermediate_out is enable, so the "
                            "IntermediateOut should not be empty."));

      auto intermediate_out = ctx.Output<framework::Tensor>("IntermediateOut");
      outputs.emplace_back(intermediate_out);
    } else {
      outputs.emplace_back(nullptr);
    }

    RunFunctors<DeviceContext, T>(ctx, in_x, in_y, &outputs);
  }
};

template <typename DeviceContext, typename T>
class FusedElemwiseAddActivationGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_y = ctx.Input<framework::Tensor>("Y");
    PADDLE_ENFORCE_NE(in_y, nullptr, platform::errors::InvalidArgument(
                                         "Input(Y) should not be nullptr."));
    auto in_out = ctx.Input<framework::Tensor>("Out");
    PADDLE_ENFORCE_NE(
        in_out, nullptr,
        platform::errors::InvalidArgument("Input(Out) should not be nullptr."));
    auto in_out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_NE(in_out_grad, nullptr,
                      platform::errors::InvalidArgument(
                          "Input(Out@Grad) should not be nullptr."));

    framework::Tensor *in_x =
        const_cast<framework::Tensor *>(ctx.Input<framework::Tensor>("X"));
    framework::Tensor *x_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    framework::Tensor *y_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    framework::Tensor *d_intermediate_out = ctx.Output<framework::Tensor>(
        framework::GradVarName("IntermediateOut"));

    auto functor_list = ctx.Attr<std::vector<std::string>>("functor_list");

    // Get intermediate_out
    framework::Tensor *in_intermediate_out = nullptr;
    if (ctx.Attr<bool>("save_intermediate_out")) {
      // if save_intermediate_out is true, for Unary(Binary(x, y)) and
      // Binary(x, Unary(y)), the Binary(x, y) and Unary(y) not need to
      // recompute.
      in_intermediate_out = const_cast<framework::Tensor *>(
          ctx.Input<framework::Tensor>("IntermediateOut"));
      PADDLE_ENFORCE_NE(in_intermediate_out, nullptr,
                        platform::errors::InvalidArgument(
                            "The option of 'save_intermediate_out' is opened,"
                            " so the number of 'Out' should be two."));
    } else {
      if (!InputXCanBeAbsent(functor_list)) {
        PADDLE_ENFORCE_NE(in_x, nullptr, platform::errors::InvalidArgument(
                                             "Input(X) should not be null."));
      }
    }

    // Get in_x
    if (ctx.HasInput("X")) {
      PADDLE_ENFORCE_NE(in_x, nullptr, platform::errors::InvalidArgument(
                                           "Input(X) should not be null."));
    } else {
      // If functor_list contains elementwise_add, the backward doesn't use
      // in_x, in_y and in_out.
      PADDLE_ENFORCE_EQ(InputXCanBeAbsent(functor_list), true,
                        platform::errors::InvalidArgument(
                            "Only when the compoundfunctor contains "
                            "elementwise_add_grad, the 'X' could be absent."));
      in_x = const_cast<framework::Tensor *>(in_out_grad);
    }

    bool has_in_place = HasInPlaceUnary(functor_list);
    if (has_in_place) {
      RunGradFunctors<DeviceContext, T, true /*InPlace*/>(
          ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, x_grad,
          y_grad, d_intermediate_out);
    } else {
      RunGradFunctors<DeviceContext, T, false /*InPlace*/>(
          ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, x_grad,
          y_grad, d_intermediate_out);
    }
  }
};
}  // namespace operators
}  // namespace paddle
