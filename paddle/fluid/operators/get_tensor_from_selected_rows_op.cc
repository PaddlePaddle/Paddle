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
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "GetTensorFromSelectedRows");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "GetTensorFromSelectedRows");

    PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("X").front(),
                      framework::proto::VarType::SELECTED_ROWS,
                      common::errors::InvalidArgument(
                          "The input X(%s)'s type should be SelectedRows, "
                          "but the received is %s",
                          ctx->Inputs("X").front(),
                          ctx->GetInputsVarType("X").front()));
    PADDLE_ENFORCE_EQ(
        ctx->GetOutputsVarType("Out").front(),
        framework::proto::VarType::DENSE_TENSOR,
        common::errors::InvalidArgument(
            "The output Out(%s)'s type should be phi::DenseTensor, "
            "but the received is %s",
            ctx->Outputs("Out").front(),
            ctx->GetOutputsVarType("Out").front()));
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }
};

template <typename T, typename DeviceContext>
class GetTensorFromSelectedRowsKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.Input<phi::SelectedRows>("X");
    auto *out = ctx.Output<phi::DenseTensor>("Out");

    out->Resize(x->value().dims());
    out->mutable_data(ctx.GetPlace(), x->value().type());
    phi::Copy(ctx.device_context(), x->value(), ctx.GetPlace(), false, out);
  }
};

class GetTensorFromSelectedRowsOpProtoMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input type is SelectedRows.");
    AddOutput("Out", "The output type is phi::DenseTensor.");
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
    ctx->SetOutputType("Out", framework::proto::VarType::DENSE_TENSOR);
    ctx->SetOutputDataType("Out", ctx->GetInputDataType("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(get_tensor_from_selected_rows,
                  ops::GetTensorFromSelectedRowsOp,
                  ops::GetTensorFromSelectedRowsOpProtoMaker,
                  ops::GetTensorFromSelectedRowsOpVarTypeInference);
