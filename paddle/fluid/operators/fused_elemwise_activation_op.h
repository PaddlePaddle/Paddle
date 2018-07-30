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
#include "paddle/fluid/operators/math/functors.h"

namespace math = paddle::operators::math;

namespace paddle {
namespace operators {

class FusedElemwiseActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

class FusedElemwiseActivationMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class FusedElemwiseActivationGradMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *op_desc_ptr = new framework::OpDesc();
    op_desc_ptr->SetType(this->ForwardOpType() + "_grad");

    for (auto &input_param : this->InputNames()) {
      op_desc_ptr->SetInput(input_param, this->Input(input_param));
      op_desc_ptr->SetOutput(framework::GradVarName(input_param),
                             this->InputGrad(input_param, true));
    }

    for (auto &output_param : this->OutputNames()) {
      op_desc_ptr->SetInput(output_param, this->Output(output_param));
      op_desc_ptr->SetInput(framework::GradVarName(output_param),
                            this->OutputGrad(output_param));
    }
    op_desc_ptr->SetAttrMap(this->Attrs());

    std::string functor_names =
        boost::get<std::string>(op_desc_ptr->GetAttr("functor_list"));
    size_t pos = functor_names.find(",");
    //    PADDLE_ENFORCE(pos != functor_names.end());
    auto func_1 = functor_names.substr(0, pos);
    auto func_2 = functor_names.substr(pos + 1, functor_names.size());

    op_desc_ptr->SetAttr("functor_list", func_1 + "_grad," + func_2 + "_grad");

    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

class FusedElemwiseActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FusedElemwiseActivationKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *in_y = ctx.Input<Tensor>("Y");
    Tensor *output = ctx.Output<Tensor>("Out");

    std::string functors = ctx.Attr<std::string>("functor_list");

    math::RunFunctors<DeviceContext, T>(ctx, functors, in_x, in_y, output);
  }
};

template <typename DeviceContext, typename T>
class FusedElemwiseActivationGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *in_y = ctx.Input<Tensor>("Y");
    const Tensor *in_out = ctx.Input<Tensor>("Out");
    const Tensor *in_out_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));

    Tensor *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor *y_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));

    std::string functors = ctx.Attr<std::string>("functor_list");

    math::RunGradFunctors<DeviceContext, T>(ctx, functors, in_x, in_y, in_out,
                                            in_out_grad, x_grad, y_grad);
  }
};

}  // namespace operators
}  // namespace paddle
