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

#include "paddle/fluid/operators/rrelu_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

using framework::Tensor;

class RReluOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "RRelu");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->SetOutputDim("Mask", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "Seed") {
      VLOG(10) << "var_name:" << var_name
               << " does not need to transform in rrelu op";
      return expected_kernel_type;
    }

    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class RReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of rrelu op.");
    AddInput("Seed",
             "The seed of rrelu op, it has higher priority than the attr "
             "fix_seed and seed")
        .AsDispensable()
        .AsExtra();
    AddOutput("Out", "The output of rrelu op.");
    AddOutput("Mask", 
              "The tensor of derivatives corresponding to each element of X.")
        .AsIntermediate()
        .AsExtra();
    AddAttr<float>("lower_bound", "Lower bound of the uniform distribution.")
        .SetDefault(.125f)
        .AddCustomChecker([](const float& lower) {
          PADDLE_ENFORCE_EQ(lower >= 0.0f && lower < 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'lower_bound' must be between 0.0 and 1.0."));
        });
    AddAttr<float>("upper_bound", "Upper bound of the uniform distribution.")
        .SetDefault(.333f)
        .AddCustomChecker([](const float& upper) {
          PADDLE_ENFORCE_EQ(upper >= 0.0f && upper < 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'upper_bound' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<int>("seed", "RRelu random seed.").SetDefault(0).AsExtra();
    AddComment(R"DOC(
RRelu Operator.
)DOC");
  }
};

class RReluOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "RReluGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "RReluGrad");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    ctx->ShareLoD(framework::GradVarName("Out"),
                  /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class RReluGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("rrelu_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Mask", this->Output("Mask"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(rrelu, ops::RReluOp, ops::RReluOpMaker,
                  ops::RReluGradOpMaker<paddle::framework::OpDesc>,
                  ops::RReluGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(rrelu_grad, ops::RReluOpGrad);
REGISTER_OP_CPU_KERNEL(
    rrelu, ops::CPURReluKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPURReluKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    rrelu_grad,
    ops::RReluGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RReluGradKernel<paddle::platform::CPUDeviceContext, double>);
