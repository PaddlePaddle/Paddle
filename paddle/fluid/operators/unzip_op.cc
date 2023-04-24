/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/unzip_op.h"

#include <memory>

#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

class unzipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "lod");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "lod");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        2UL,
        platform::errors::InvalidArgument(
            "Input(X)'s rank should be 2, but got %d", x_dims.size()));

    auto lod_dims = ctx->GetInputDim("lod");
    PADDLE_ENFORCE_EQ(
        lod_dims.size(),
        1UL,
        platform::errors::InvalidArgument(
            "Input(X)'s rank should be 1, but got %d", lod_dims.size()));

    ctx->SetOutputDim("Y", {lod_dims[0] - 1, x_dims[1]});
  }

 protected:
  // Explicitly set that the data type of computation kernel of
  // unzip
  // is determined by its input "X".
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }
};

class unzipGradientOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "unzipGradient");
    OP_INOUT_CHECK(ctx->HasInput("lod"), "Input", "unzip", "unzipGradient");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   framework::GradVarName("Y"),
                   "unzipGradient");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "unzipGradient");

    auto x_dims = ctx->GetInputDim("X");
    auto lod_dims = ctx->GetInputDim("lod");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Expect Input(X)'s rank == 2, but got %d", x_dims.size()));
    PADDLE_ENFORCE_EQ(
        dy_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Expect Input(X)'s rank == 2, but got %d", dy_dims.size()));
    PADDLE_ENFORCE_EQ(
        lod_dims.size(),
        1,
        platform::errors::InvalidArgument(
            "Expect Input(X)'s rank == 1, but got %d", lod_dims.size()));

    PADDLE_ENFORCE_EQ(
        x_dims[1],
        dy_dims[1],
        platform::errors::InvalidArgument(
            "The 1st dimension of Input(X) and Input(Y@Grad) should "
            "be equal, X is %d, Y@Grad is %d",
            x_dims[1],
            dy_dims[1]));

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of
  // unzip
  // is determined by its input "X".
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Y")),
                          ctx.device_context().GetPlace());
  }
};

class unzipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LodTensor, default LodTensor<float>), a 2-D tensor with shape "
             "[M x N],"
             " where N is the batch size and D is the emebdding dim. ");
    AddInput("lod", "(Tensor),  a 1-D Tensor with shape [K]");
    AddOutput("Y",
              "(LodTensor, default LodTensor<float>), a 2-D tensor with shape "
              "[K-1 x N].");
    AddComment(R"DOC(
unzip Operator.
)DOC");
  }
};

template <typename T>
class unzipGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("unzip_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("lod", this->Input("lod"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(unzip,
                  ops::unzipOp,
                  ops::unzipOpMaker,
                  ops::unzipGradOpMaker<paddle::framework::OpDesc>,
                  ops::unzipGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(unzip_grad, ops::unzipGradientOp);

PD_REGISTER_STRUCT_KERNEL(unzip, CPU, ALL_LAYOUT, ops::unzipOpKernel, int64_t) {
}
PD_REGISTER_STRUCT_KERNEL(
    unzip_grad, CPU, ALL_LAYOUT, ops::unzipGradOpKernel, int64_t) {}
