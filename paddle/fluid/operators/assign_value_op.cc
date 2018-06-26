//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/assign_value_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class AssignValueOp : public framework::OperatorWithKernel {
 public:
  AssignValueOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of AssignValueOp should not be null.");
    auto shape = ctx->Attrs().Get<std::vector<int>>("shape");
    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class AssignValueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "(Tensor) Output tensor of assign_value operator.");
    AddAttr<std::vector<int>>("shape",
                              "(vector<int>) "
                              "Shape of values.");
    AddAttr<int>("dtype", "data type of values")
        .InEnum({framework::proto::VarType::INT32,
                 framework::proto::VarType::FP32});
    AddAttr<std::vector<float>>("fp32_values", "store the float values")
        .SetDefault({});
    AddAttr<std::vector<int>>("int32_values", "store the int values")
        .SetDefault({});
    AddComment(R"DOC(
AssignValue operator

$$Out = values$$
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(assign_value, ops::AssignValueOp, ops::AssignValueOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(assign_value, ops::AssignValueKernel<int>,
                       ops::AssignValueKernel<float>);
