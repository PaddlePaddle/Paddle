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

#include "paddle/fluid/operators/optimizers/sgd_op.h"
#include <string>
namespace paddle {
namespace operators {

class SGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of SGDOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 element");
    auto param_dim = ctx->GetInputDim("Param");
    // TODO(qijun): check dimensions of Param and Grad at compile
    // and runtime.
    ctx->SetOutputDim("ParamOut", param_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Param"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (var_name == "LearningRate") {
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class SGDOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto &input_var_n = ctx->Input("Param")[0];
    auto in_var_type = ctx->GetType(input_var_n);
    PADDLE_ENFORCE(in_var_type == framework::proto::VarType::SELECTED_ROWS ||
                       in_var_type == framework::proto::VarType::LOD_TENSOR,
                   "The input Var's type should be LoDtensor or SelectedRows,"
                   " but the received var(%s)'s type is %s",
                   input_var_n, in_var_type);

    for (auto &out_var_n : ctx->Output("ParamOut")) {
      if (ctx->GetType(out_var_n) != in_var_type) {
        ctx->SetType(out_var_n, in_var_type);
      }
    }
  }
};

class SGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor or SelectedRows) Input parameter");
    AddInput("LearningRate", "(Tensor) Learning rate of SGD");
    AddInput("Grad", "(Tensor or SelectedRows) Input gradient");
    AddOutput("ParamOut",
              "(Tensor or SelectedRows, same with Param) "
              "Output parameter, should share the same memory with Param");
    AddComment(R"DOC(

SGD operator

This operator implements one step of the stochastic gradient descent algorithm.

$$param\_out = param - learning\_rate * grad$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sgd, ops::SGDOp, ops::SGDOpMaker,
                  paddle::framework::EmptyGradOpMaker, ops::SGDOpInferVarType);
REGISTER_OP_CPU_KERNEL(sgd, ops::SGDOpKernel<float>, ops::SGDOpKernel<double>);
