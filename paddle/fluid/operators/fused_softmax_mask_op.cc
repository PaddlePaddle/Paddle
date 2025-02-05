/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle::operators {

class SoftmaxMaskFuseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class SoftmaxMaskFuseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of softmax_mask_fuse op, "
             "which is the result of matmul(QK)/sqrt(dk).");
    AddInput("Mask", "The mask attr of the op, multi-head attention's mask");
    AddOutput("Out", "The result of softmax_mask_fuse op.");

    AddComment(R"DOC(
Softmax Mask Fuse Operator.
In general, the compute pass is:
product = matmul(QK)/sqrt(dk)
pre_softmax = product + attn_mask
output = softmax(pre_softmax)
To reduce the launch op time and reduce the number of forward and backward,
and to reduce the memory cost for the pre_softmax var during the compute
this op fuse last two operations into one, so users can simply call
product = matmul(QK)/sqrt(dk)
output = softmax_mask_fuse(product, attn_mask)
to get the final output.
By doing this fusion, we can optimize the training by
1. saving one launch cost, one forward and one backward cost
2. saving the memory cost used to save the tmp var
)DOC");
  }
};

class SoftmaxMaskFuseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class SoftmaxMaskFuseGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_softmax_mask_grad");
    op->SetInput("Softmax", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(fused_softmax_mask,
                            SoftmaxMaskFuseInferShapeFunctor,
                            PD_INFER_META(phi::SoftmaxMaskFuseInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(fused_softmax_mask_grad,
                            SoftmaxMaskFuseGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralUnaryGradInferMeta));
REGISTER_OPERATOR(fused_softmax_mask,
                  ops::SoftmaxMaskFuseOp,
                  ops::SoftmaxMaskFuseOpMaker,
                  ops::SoftmaxMaskFuseGradOpMaker<paddle::framework::OpDesc>,
                  ops::SoftmaxMaskFuseGradOpMaker<paddle::imperative::OpBase>,
                  SoftmaxMaskFuseInferShapeFunctor);
REGISTER_OPERATOR(fused_softmax_mask_grad,
                  ops::SoftmaxMaskFuseOpGrad,
                  SoftmaxMaskFuseGradInferShapeFunctor);
