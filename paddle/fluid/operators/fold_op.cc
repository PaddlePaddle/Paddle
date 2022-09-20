/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class FoldOp : public framework::OperatorWithKernel {
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

class FoldOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, "
             "the input of fold op. "
             "The format of X is [N, C_in, L], "
             "where N is the batch size, C_in is the input channels, "
             "L is the length");
    AddOutput("Y",
              "Tensor, "
              "the output of unfold op. "
              "The format of Y is [N, C_out, output_height, output_width], "
              "where N is the batch size, "
              "C_in is the output channels of Y, output_height and "
              "output_width "
              "is the calculated height and width of output feature map.");
    AddAttr<std::vector<int>>(
        "output_sizes",
        "vector<int>, the output sizes of the convolution operator.");
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
**Fold Operator**

This Operator is used to combines an array of sliding local blocks into a large containing
tensor. also known as col2im when operated on batched 2D image tensor. Fold calculates each
combined value in the resulting large tensor by summing all values from all containing blocks.
Unfold extracts the values in the local blocks by copying from the large tensor. So, if the
blocks overlap, they are not inverses of each other.
    )DOC");
  }
};

class FoldGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }
};

template <typename T>
class FoldGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fold_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetInput("X", this->Input("X"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(FoldGradOpNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(fold,
                            FoldInferShapeFunctor,
                            PD_INFER_META(phi::FoldInferMeta));
REGISTER_OPERATOR(fold,
                  ops::FoldOp,
                  ops::FoldOpMaker,
                  ops::FoldGradMaker<paddle::framework::OpDesc>,
                  ops::FoldGradMaker<paddle::imperative::OpBase>,
                  FoldInferShapeFunctor);
DECLARE_INFER_SHAPE_FUNCTOR(fold_grad,
                            FoldGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));
REGISTER_OPERATOR(fold_grad,
                  ops::FoldGradOp,
                  ops::FoldGradOpNoNeedBufferVarsInferer,
                  FoldGradInferShapeFunctor);
