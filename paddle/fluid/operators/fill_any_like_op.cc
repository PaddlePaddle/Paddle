/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class FillAnyLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fill_any_like");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "fill_any_like");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    const auto &data_type = ctx.Attr<int>("dtype");
    if (data_type >= 0) {
      kt.data_type_ = static_cast<framework::proto::VarType::Type>(data_type);
    }
    return kt;
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class FillAnyLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of fill-zeros-like op.");
    AddOutput("Out", "The variable will be filled up with specified value.");
    AddAttr<float>("value", "The filled value").SetDefault(0.0);
    AddAttr<int>("dtype",
                 "Output tensor data type. default value is -1,"
                 "according to the input dtype.")
        .SetDefault(-1);
    AddComment(R"DOC(
FillAnyLike Operator.

Fill up a variable with Attr(value).
The output will have the same shape and dtype as the input.

)DOC");
  }
};

class FillAnyLikeVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_data_type = static_cast<framework::proto::VarType::Type>(
        PADDLE_GET_CONST(int, ctx->GetAttr("dtype")));
    if (var_data_type < 0) {
      ctx->SetOutputDataType("Out", ctx->GetInputDataType("X"));
    } else {
      ctx->SetOutputDataType("Out", var_data_type);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fill_any_like,
    ops::FillAnyLikeOp,
    ops::FillAnyLikeOpMaker,
    ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::FillAnyLikeVarTypeInference)
