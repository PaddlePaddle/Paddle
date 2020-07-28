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

#include "paddle/fluid/operators/amp/amp_check_finite_and_scale_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class AmpCheckFiniteAndScaleOp : public framework::OperatorWithKernel {
 public:
  AmpCheckFiniteAndScaleOp(const std::string &type,
                           const framework::VariableNameMap &inputs,
                           const framework::VariableNameMap &outputs,
                           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X",
                   "amp_check_finite_and_unscale");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out",
                   "amp_check_finite_and_unscale");
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("X").size(), ctx->Outputs("Out").size(),
        platform::errors::InvalidArgument(
            "The input(X) and output(Out) should have same size in "
            "Operator(amp_check_finite_and_unscale), size of input(X) is %d "
            "and size of output(Out) is %d.",
            ctx->Inputs("X").size(), ctx->Outputs("Out").size()));
    auto x_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim("Out", x_dims);
    ctx->SetOutputDim("FoundInfinite", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class AmpCheckFiniteAndScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensors) The input tensors of amp_check_finite_and_scale operator.")
        .AsDuplicable();
    AddInput("Scale",
             "(Tensor) 1-dim tensor, the scale of amp_check_finite_and_scale "
             "operator.");
    AddOutput("Out",
              "(Tensors) The scaled output tensor of "
              "amp_check_finite_and_unscale operator.")
        .AsDuplicable();
    AddOutput("FoundInfinite",
              "(Tensor) 1-dim tensor, contains a int scalar, which indicates "
              "if there there is infinite or nan item in input X.");
    AddComment(R"DOC(
amp_check_finite_and_scale operator.
Check if input X contains all finite data, if yes, scale it by input Scale.

$$Out = X * scale$$

If any tensor in X contains Inf or Nan, the Out will generate a indicator.
FoundInfinite will be 1 (True), and Out will not be scaled. In this case, the data of 
Out should not be used, and its data may not be deterministic. 
Otherwise, FoundInfinite will be 0 (False).

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    amp_check_finite_and_scale, ops::AmpCheckFiniteAndScaleOp,
    ops::AmpCheckFiniteAndScaleOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    amp_check_finite_and_scale,
    ops::AmpCheckFiniteAndScaleKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    ops::AmpCheckFiniteAndScaleKernel<paddle::platform::CPUDeviceContext,
                                      double>);
