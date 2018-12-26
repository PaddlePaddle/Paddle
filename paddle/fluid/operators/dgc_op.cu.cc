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

#include "paddle/fluid/operators/dgc_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class DGCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    /*
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
  */
  }

 protected:
  /*
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Param"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
  */
};

class DGCOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    /*
   auto input_var_n = op_desc.Input("Param")[0];
   auto in_var_type = block->FindRecursiveOrCreateVar(input_var_n).GetType();
   PADDLE_ENFORCE(in_var_type == framework::proto::VarType::SELECTED_ROWS ||
                      in_var_type == framework::proto::VarType::LOD_TENSOR,
                  "The input Var's type should be LoDtensor or SelectedRows,"
                  " but the received var(%s)'s type is %s",
                  input_var_n, in_var_type);

   for (auto &out_var_n : op_desc.Output("ParamOut")) {
     auto &out_var = block->FindRecursiveOrCreateVar(out_var_n);
     if (out_var.GetType() != in_var_type) {
       out_var.SetType(in_var_type);
     }
   }
 */
  }
};

class DGCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("U", "(Tensor) Middle tensor of DGC");
    AddInput("V", "(Tensor) Middle tensor of DGC");
    AddInput("m", "(Scalar) Momentum correction parameter.");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("GradLocal", "(Tensor) Local gradient for accumulation.");
    AddOutput("EncodeGradient",
              "(Tensor) "
              "Output encoded gradient");
    AddComment(R"DOC(
    Please see appendix D of https://arxiv.org/abs/1712.01887.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(dgc, ops::DGCOp, ops::DGCOpMaker);

REGISTER_OP_CUDA_KERNEL(
    dgc, ops::DGCOpKernel<paddle::platform::CUDADeviceContext, float>);
