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

#include "paddle/fluid/operators/cross_entropy2_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/operators/cross_entropy_op_base.h"

namespace paddle {
namespace operators {

class CrossEntropyOp2 : public CrossEntropyOpBase {
 public:
  using CrossEntropyOpBase::CrossEntropyOpBase;

  void InferShape(framework::InferShapeContext* ctx) const override {
    CrossEntropyOpBase::InferShape(ctx);

    PADDLE_ENFORCE(ctx->HasOutput("XShape"),
                   "Output(XShape) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto x_dims_vec = framework::vectorize(x_dims);
    x_dims_vec.push_back(0);
    ctx->SetOutputDim("XShape", framework::make_ddim(x_dims_vec));
    ctx->ShareLoD("X", /*->*/ "XShape");
  }

 protected:
  bool IsSoftLabel(framework::InferShapeContext* ctx) const override {
    return false;
  }
};

class CrossEntropyGradientOp2 : public CrossEntropyGradientOpBase {
 public:
  using CrossEntropyGradientOpBase::CrossEntropyGradientOpBase;

 protected:
  virtual framework::DDim GetXDim(framework::InferShapeContext* ctx) const {
    auto x_shape = ctx->GetInputDim("XShape");
    return framework::DDim(x_shape.Get(), x_shape.size() - 1);
  }

  virtual const char* VarNameWithXLoD() const { return "XShape"; }

  virtual bool IsSoftLabel(framework::InferShapeContext* ctx) const {
    return false;
  }
};

class CrossEntropyOpMaker2 : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a tensor whose last dimension "
             "size is equal to the number of classes. This input is a "
             "probability computed by the previous operator, which is almost "
             "always the result of a softmax operator.");
    AddInput(
        "Label",
        "(Tensor), the tensor which represents the ground truth. It has the "
        "same shape with 'X' except the last dimension. One hot Tensor.");
    AddOutput("Y",
              "(Tensor, default Tensor<float>), a tensor whose shape is same "
              "with 'X' except that the last dimension size is 1. It "
              "represents the cross entropy loss.");
    AddOutput("XShape", "Temporaily variable to save shape and LoD of X.");
    AddAttr<int>("ignore_index",
                 "(int, default -100), Specifies a target value that is"
                 "ignored and does not contribute to the input gradient."
                 "Only valid if soft_label is set to False")
        .SetDefault(-100);
    AddComment(R"DOC(
Hard-label CrossEntropy Operator.

The input 'X' and 'Label' will first be logically flattened to 2-D matrixs. 
The matrix's second dimension(row length) is as same as the original last 
dimension, and the first dimension(column length) is the product of all other 
original dimensions. Then the softmax computation will take palce on each raw 
of flattened matrixs.

Only support hard label.

Both the input X and Label can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input X.

)DOC");
  }
};

class CrossEntropyGradOpMaker2 : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("cross_entropy_grad2");
    op->SetInput("Label", Input("Label"));
    op->SetInput("Y", Output("Y"));
    op->SetInput("XShape", Output("XShape"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPUCtx = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(cross_entropy2, ops::CrossEntropyOp2,
                  ops::CrossEntropyOpMaker2, ops::CrossEntropyOpInferVarType,
                  ops::CrossEntropyGradOpMaker2);
REGISTER_OPERATOR(cross_entropy_grad2, ops::CrossEntropyGradientOp2);
REGISTER_OP_CPU_KERNEL(cross_entropy2,
                       ops::CrossEntropyOpKernel2<CPUCtx, float>,
                       ops::CrossEntropyOpKernel2<CPUCtx, double>);
REGISTER_OP_CPU_KERNEL(cross_entropy_grad2,
                       ops::CrossEntropyGradientOpKernel2<CPUCtx, float>,
                       ops::CrossEntropyGradientOpKernel2<CPUCtx, double>);
