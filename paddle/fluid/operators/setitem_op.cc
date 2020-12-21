//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/setitem_op.h"

#include <string>

namespace paddle {
namespace operators {

class Setitem : public framework::OperatorWithKernel {
 public:
  Setitem(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Setitem");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Setitem");
    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_LT(
        in_dims.size(), 7,
        platform::errors::InvalidArgument(
            "The rank of input should be less than 7, but received %d.",
            in_dims.size()));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class SetitemMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) Input tensor of setitem operator.");
    AddInput("ValueTensor", "(Tensor) Value tensor of setitem operator.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor) Output tensor of setitem operator. The output is the "
              "same Tensor as input");

    AddAttr<int>("dtype", "data type of input.")
        .InEnum(
            {framework::proto::VarType::BOOL, framework::proto::VarType::INT32,
             framework::proto::VarType::INT64, framework::proto::VarType::FP32,
             framework::proto::VarType::FP64})
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<std::vector<int>>(
        "axes", "(list<int>) Axes that `starts` and `ends` apply to.");
    AddAttr<std::vector<int>>(
        "starts",
        "(list<int>) Starting indices of corresponding axis in `axes`");
    AddAttr<std::vector<int>>(
        "ends", "(list<int>) Ending indices of corresponding axis in `axes`.");

    AddAttr<std::vector<int>>("bool_values", "store the bool values")
        .SetDefault({});
    AddAttr<std::vector<float>>("fp32_values", "store the float32 values")
        .SetDefault({});
    AddAttr<std::vector<int>>("int32_values", "store the int32 values")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("int64_values", "store the int64 values")
        .SetDefault({});

    AddAttr<std::vector<int>>("shape", "(vector<int>) Shape of values.")
        .SetDefault({});
    AddComment(R"DOC(Setitem operator.
Assignment to a Tensor in static mode.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    setitem, ops::Setitem, ops::SetitemMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    setitem, ops::SetitemKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SetitemKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::SetitemKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SetitemKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SetitemKernel<paddle::platform::CPUDeviceContext, bool>);
