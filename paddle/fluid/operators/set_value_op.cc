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

#include "paddle/fluid/operators/set_value_op.h"

#include <string>

namespace paddle {
namespace operators {

class SetValue : public framework::OperatorWithKernel {
 public:
  SetValue(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "SetValue");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SetValue");
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

class SetValueMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) Input tensor of set_value operator.");
    AddInput("ValueTensor", "(Tensor) Value tensor of set_value operator.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor) Output tensor of set_value operator. The output is the "
              "same Tensor as input");

    AddAttr<int>("dtype", "data type of input.")
        .InEnum(
            {framework::proto::VarType::BOOL, framework::proto::VarType::INT32,
             framework::proto::VarType::INT64, framework::proto::VarType::FP32,
             framework::proto::VarType::FP64})
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<std::vector<int64_t>>(
        "axes", "(list<int64_t>) Axes that `starts` and `ends` apply to.");
    AddAttr<std::vector<int64_t>>(
        "starts",
        "(list<int64_t>) Starting indices of corresponding axis in `axes`");
    AddAttr<std::vector<int64_t>>(
        "ends",
        "(list<int64_t>) Ending indices of corresponding axis in `axes`.");

    AddAttr<std::vector<int>>("bool_values", "store the bool values")
        .SetDefault({});
    AddAttr<std::vector<float>>("fp32_values", "store the float32 values")
        .SetDefault({});
    AddAttr<std::vector<int>>("int32_values", "store the int32 values")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("int64_values", "store the int64 values")
        .SetDefault({});
    AddAttr<std::vector<double>>("fp64_values", "store the float64 values")
        .SetDefault({});

    AddAttr<std::vector<int64_t>>("shape", "(vector<int64_t>) Shape of values.")
        .SetDefault({});
    AddComment(R"DOC(SetValue operator.
Assignment to a Tensor in static mode.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    set_value, ops::SetValue, ops::SetValueMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    set_value, ops::SetValueKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SetValueKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::SetValueKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SetValueKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SetValueKernel<paddle::platform::CPUDeviceContext, bool>);
