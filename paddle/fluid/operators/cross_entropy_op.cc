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

#include "paddle/fluid/operators/cross_entropy_op.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {

class CrossEntropyOpBase : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(rank, label_dims.size(),
                      "Input(X) and Input(Label) shall have the same rank.");
    bool has_mutable_dim = framework::has_mutable_dim(x_dims)  || framework::has_mutable_dim(label_dims);
    bool check = ctx->IsRuntime() || !has_mutable_dim;
    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(label_dims, 0, rank - 1),
                        "Input(X) and Input(Label) shall have the same shape "
                        "except the last dimension.");
    }

    if (IsSoftLabel(ctx)) {
      if (check) {
        PADDLE_ENFORCE_EQ(x_dims[rank - 1], label_dims[rank - 1],
                          "If Attr(soft_label) == true, the last dimension of "
                          "Input(X) and Input(Label) should be equal.");
      }
    } else {
      PADDLE_ENFORCE_EQ(label_dims[rank - 1], 1UL,
                        "If Attr(softLabel) == false, the last dimension of "
                        "Input(Label) should be 1.");
    }

    auto y_dims = x_dims;
    y_dims[rank - 1] = 1;
    ctx->SetOutputDim("Y", y_dims);
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }

  virtual bool IsSoftLabel(framework::InferShapeContext* ctx) const {
    return ctx->Attrs().Get<bool>("soft_label");
  }
};

class CrossEntropyGradientOpBase : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) shoudl be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = GetXDim(ctx);
    auto label_dims = ctx->GetInputDim("Label");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(dy_dims.size(), rank,
                      "Input(Y@Grad) and Input(X) should have the same rank.");
    PADDLE_ENFORCE_EQ(label_dims.size(), rank,
                      "Input(Label) and Input(X) should have the same rank.");

    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(label_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(label_dims, 0, rank - 1),
                        "The Input(X) and Input(Label) should have the same "
                        "shape except the last dimension.");
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(dy_dims, 0, rank - 1),
                        "The Input(X) and Input(Y@Grad) should have the same "
                        "shape except the last dimension.");
    }
    if (IsSoftLabel(ctx)) {
      if (check) {
        PADDLE_ENFORCE_EQ(
            x_dims[rank - 1], label_dims[rank - 1],
            "When Attr(soft_label) == true, the last dimension of "
            "Input(X) and Input(Label) should be equal.");
      }
    } else {
      PADDLE_ENFORCE_EQ(label_dims[rank - 1], 1,
                        "When Attr(soft_label) == false, the last dimension of "
                        "Input(Label) should be 1.");
    }
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    PADDLE_ENFORCE_EQ(dy_dims[rank - 1], 1,
                      "The last dimension of Input(Y@Grad) should be 1.");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD(VarNameWithXLoD(), framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Y"))->type(),
        ctx.device_context());
  }

  virtual framework::DDim GetXDim(framework::InferShapeContext* ctx) const {
    return ctx->GetInputDim("X");
  }

  virtual const char* VarNameWithXLoD() const { return "X"; }

  virtual bool IsSoftLabel(framework::InferShapeContext* ctx) const {
    return ctx->Attrs().Get<bool>("soft_label");
  }
};

class CrossEntropyOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Y"}};
  }
};

class CrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
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
        "same shape with 'X' except the last dimension. When soft_label is set "
        "to false, the last dimension size is 1; when soft_label is set to "
        "true, the last dimension size is equal to the number of classes.");
    AddOutput("Y",
              "(Tensor, default Tensor<float>), a tensor whose shape is same "
              "with 'X' except that the last dimension size is 1. It "
              "represents the cross entropy loss.");
    AddAttr<bool>("soft_label",
                  "(bool, default false), a flag indicating whether to "
                  "interpretate the given labels as soft labels.")
        .SetDefault(false);
    AddAttr<int>("ignore_index",
                 "(int, default -100), Specifies a target value that is"
                 "ignored and does not contribute to the input gradient."
                 "Only valid if soft_label is set to False")
        .SetDefault(-100);
    AddComment(R"DOC(
CrossEntropy Operator.

The input 'X' and 'Label' will first be logically flattened to 2-D matrixs. 
The matrix's second dimension(row length) is as same as the original last 
dimension, and the first dimension(column length) is the product of all other 
original dimensions. Then the softmax computation will take palce on each raw 
of flattened matrixs.

It supports both standard cross-entropy and soft-label cross-entropy loss
computation.
1) One-hot cross-entropy:
    soft_label = false, Label[i, 0] indicates the class index for sample i:

                $Y[i] = -\log(X[i, Label[i]])$

2) Soft-label cross-entropy:
    soft_label = true, Label[i, j] indicates the soft label of class j
    for sample i:

                $Y[i] = \sum_j{-Label[i, j] * log(X[i, j])}$

   Please make sure that in this case the summuation of each row of Label
   equals one.

3) One-hot cross-entropy with vecterized Input(Label):
     As a special case of 2), when each row of Input(Label) has only one
     non-zero element (equals 1), soft-label cross-entropy degenerates to a
     one-hot cross-entropy with one-hot label representation.

Both the input X and Label can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input X.

)DOC");
  }
};

class CrossEntropyGradientOp : public CrossEntropyGradientOpBase {
 public:
  using CrossEntropyGradientOpBase::CrossEntropyGradientOpBase;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    CrossEntropyGradientOpBase::InferShape(ctx);
  }
};

class CrossEntropyGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("cross_entropy_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Label", Input("Label"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

class CrossEntropyOp2 : public CrossEntropyOpBase {
 public:
  using CrossEntropyOpBase::CrossEntropyOpBase;

  void InferShape(framework::InferShapeContext* ctx) const override {
    CrossEntropyOpBase::InferShape(ctx);

    PADDLE_ENFORCE(ctx->HasOutput("XShape"),
                   "Output(XShape) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("MatchX"),
                   "Output(MatchX) should be not null.");
    auto x_dims = ctx->GetInputDim("X");
    auto x_dims_vec = framework::vectorize(x_dims);
    x_dims_vec.push_back(0);
    ctx->SetOutputDim("XShape", framework::make_ddim(x_dims_vec));
    x_dims[x_dims.size() - 1] = 1;
    ctx->SetOutputDim("MatchX", x_dims);
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
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("MatchX"), "Input(MatchX) must exist");
    CrossEntropyGradientOpBase::InferShape(ctx);
  }

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
    AddOutput("MatchX",
              "X value that matches label, used for gradient computation.");
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

class CrossEntropyGradOpDescMaker2 : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("cross_entropy_grad2");
    op->SetInput("Label", Input("Label"));
    op->SetInput("MatchX", Output("MatchX"));
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

REGISTER_OPERATOR(cross_entropy, ops::CrossEntropyOpBase,
                  ops::CrossEntropyOpMaker, ops::CrossEntropyOpInferVarType,
                  ops::CrossEntropyGradOpDescMaker);
REGISTER_OPERATOR(cross_entropy_grad, ops::CrossEntropyGradientOp);
REGISTER_OP_CPU_KERNEL(cross_entropy, ops::CrossEntropyOpKernel<CPUCtx, float>,
                       ops::CrossEntropyOpKernel<CPUCtx, double>);
REGISTER_OP_CPU_KERNEL(cross_entropy_grad,
                       ops::CrossEntropyGradientOpKernel<CPUCtx, float>,
                       ops::CrossEntropyGradientOpKernel<CPUCtx, double>);

REGISTER_OPERATOR(cross_entropy2, ops::CrossEntropyOp2,
                  ops::CrossEntropyOpMaker2, ops::CrossEntropyOpInferVarType,
                  ops::CrossEntropyGradOpDescMaker2);
REGISTER_OPERATOR(cross_entropy_grad2, ops::CrossEntropyGradientOp2);
REGISTER_OP_CPU_KERNEL(cross_entropy2,
                       ops::CrossEntropyOpKernel2<CPUCtx, float>,
                       ops::CrossEntropyOpKernel2<CPUCtx, double>);
REGISTER_OP_CPU_KERNEL(cross_entropy_grad2,
                       ops::CrossEntropyGradientOpKernel2<CPUCtx, float>,
                       ops::CrossEntropyGradientOpKernel2<CPUCtx, double>);
