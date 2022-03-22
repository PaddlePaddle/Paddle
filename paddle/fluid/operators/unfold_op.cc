/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class UnfoldOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, "
             "the input of unfold op. "
             "The format of X is [N, C_in, H, W], "
             "where N is the batch size, C_in is the input channels, "
             "H is the height and W is the width");
    AddOutput(
        "Y",
        "Tensor, "
        "the output of unfold op. "
        "The format of Y is [N, C_in*filter_height*filter_width, "
        "output_height*output_width], where N is the batch size, "
        "C_in is the input channels of X, filter_height and filter_width is "
        "height and width of the filtering kernel, output_height and "
        "output_width "
        "is the calculated height and width of output feature map.");
    AddAttr<std::vector<int>>(
        "kernel_sizes",
        "vector<int>, the kernel sizes of the convolution operator.");
    AddAttr<std::vector<int>>(
        "strides", "vector<int>, the strides of the convolution operator.");
    AddAttr<std::vector<int>>(
        "paddings",
        "vector<int>, the paddings applied to pad the feature map.");
    AddAttr<std::vector<int>>(
        "dilations", "vector<int>, the dilations of the convolution operator.");
    AddComment(R"DOC(
**Unfold Operator**

This Operator is used to extract sliding local blocks from a batched input tensor, also known
as im2col when operated on batched 2D image tensor. For each block under the convolution filter,
all element will be rearranged as a column. While the convolution filter sliding over the input
feature map, a series of such columns will be formed. 
    )DOC");
  }
};

class UnfoldOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class UnfoldGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Y")), true,
        platform::errors::NotFound("The gradient of Y should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("The input X should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::NotFound("The gradient of X should not be null"));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }
};

template <typename T>
class UnfoldGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("unfold_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetInput("X", this->Input("X"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(UnfoldGradOpNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(unfold, UnfoldInferShapeFunctor,
                            PD_INFER_META(phi::UnfoldInferMeta));
REGISTER_OPERATOR(unfold, ops::UnfoldOp, ops::UnfoldOpMaker,
                  ops::UnfoldGradMaker<paddle::framework::OpDesc>,
                  ops::UnfoldGradMaker<paddle::imperative::OpBase>,
                  UnfoldInferShapeFunctor);
REGISTER_OPERATOR(unfold_grad, ops::UnfoldGradOp,
                  ops::UnfoldGradOpNoNeedBufferVarsInferer);
