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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace operators {

class GetTensorFromSelectedRowsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "GetTensorFromSelectedRowsOp must has input X.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "GetTensorFromSelectedRowsOp must has output Out.");
    PADDLE_ENFORCE(
        ctx->GetInputsVarType("X").front() ==
            framework::proto::VarType::SELECTED_ROWS,
        "The input X's type should be SelectedRows, but the received is %s",
        ctx->Inputs("X").front(), ctx->GetInputsVarType("X").front());
    PADDLE_ENFORCE(
        ctx->GetOutputsVarType("Out").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The output Out's type should be LoDTensor, but the received is %s",
        ctx->Outputs("Out").front(), ctx->GetOutputsVarType("Out").front());

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::GetDataTypeOfVar(ctx.InputVar("X")), ctx.device_context());
  }
};

class GetTensorFromSelectedRowsKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.Input<framework::SelectedRows>("X");
    auto *out = ctx.Output<framework::LoDTensor>("Out");

    out->Resize(x->value().dims());
    out->mutable_data(ctx.GetPlace(), x->value().type());
    framework::TensorCopy(x->value(), ctx.GetPlace(), ctx.device_context(),
                          out);
  }
};

class GetTensorFromSelectedRowsOpProtoMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input type is SelectedRows.");
    AddOutput("Out", "The output type is LoDTensor.");
    AddComment(
        R"DOC(
GetTensorFromSelectedRows Operator

GetTensorFromSelectedRows is used to get the tensor from SelectedRows.

)DOC");
  }
};

class GetTensorFromSelectedRowsOpVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const {  // NOLINT
    auto out_var_name = ctx->Output("Out").front();
    auto in_var_name = ctx->Input("X").front();

    ctx->SetType(out_var_name, framework::proto::VarType::LOD_TENSOR);
    ctx->SetDataType(out_var_name, ctx->GetDataType(in_var_name));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(get_tensor_from_selected_rows,
                  ops::GetTensorFromSelectedRowsOp,
                  ops::GetTensorFromSelectedRowsOpProtoMaker,
                  ops::GetTensorFromSelectedRowsOpVarTypeInference);

REGISTER_OP_CPU_KERNEL_FUNCTOR(get_tensor_from_selected_rows, float,
                               ops::GetTensorFromSelectedRowsKernel, double,
                               ops::GetTensorFromSelectedRowsKernel, int,
                               ops::GetTensorFromSelectedRowsKernel, int64_t,
                               ops::GetTensorFromSelectedRowsKernel);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL_FUNCTOR(get_tensor_from_selected_rows, float,
                                ops::GetTensorFromSelectedRowsKernel, double,
                                ops::GetTensorFromSelectedRowsKernel, int,
                                ops::GetTensorFromSelectedRowsKernel, int64_t,
                                ops::GetTensorFromSelectedRowsKernel);
#endif
