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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class ElementwiseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of elementwise op should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of elementwise op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of elementwise op should not be null.");

    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Y").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s [%s]",
        ctx->GetInputsVarType("Y").front(), ctx->Inputs("Y").front());

    if (ctx->GetInputsVarType("X").front() ==
        framework::proto::VarType::LOD_TENSOR) {
      auto x_dim = ctx->GetInputDim("X");
      auto y_dim = ctx->GetInputDim("Y");
      PADDLE_ENFORCE_GE(
          x_dim.size(), y_dim.size(),
          "ShapeError: the dimension of input X must greater than or equal to "
          "the one of input Y. But received: the shape of input X = [%s], the "
          "dimension of input X = %d, the shape of input Y = [%s], the "
          "dimension of input Y = %d",
          x_dim, x_dim.size(), y_dim, y_dim.size());
    } else if (ctx->GetInputsVarType("X").front() ==
               framework::proto::VarType::SELECTED_ROWS) {
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Y").size(), 1u,
          "ShapeError: For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
          "), Y must be scalar. But reveived the dimension of Y = %s",
          ctx->GetInputDim("Y").size());
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Y")[0], 1,
          "ShapeError: For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
          "), Y must be scalar. But reveived the first dimension of Y = %s",
          ctx->GetInputDim("Y")[0]);
    } else {
      PADDLE_THROW("X's type[%s] is not supported by elementwise_op.",
                   ctx->GetInputsVarType("X").front());
    }

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::GetDataTypeOfVar(ctx.InputVar("X"));

#ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ElementwiseOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};

class ElementwiseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInputX();
    AddInputY();
    AddOpOutput();

    AddAttr<int>("axis",
                 "(int, default -1). If X.dimension != Y.dimension,"
                 "Y.dimension must be a subsequence of x.dimension. And axis "
                 "is the start dimension index "
                 "for broadcasting Y onto X. ")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<bool>("use_mkldnn", "(bool, default false). Used by MKLDNN.")
        .SetDefault(false);
    AddAttr<std::string>(
        "x_data_format",
        "(string, default NCHW) Only used in mkldnn"
        "An optional string from: \"NHWC\", \"NCHW\", \"NCHW16C\", \"NCHW8C\". "
        "Defaults to \"\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("");
    AddAttr<std::string>(
        "y_data_format",
        "(string, default \"\") Only used in mkldnn"
        "An optional string from: \"NHWC\", \"NCHW\", \"NCHW16C\", \"NCHW8C\". "
        "Defaults to \"\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("");

    AddOpComment();
  }

 protected:
  virtual void AddInputX() {
    AddInput("X", "(Tensor), The first input tensor of elementwise op.");
  }
  virtual void AddInputY() {
    AddInput("Y", "(Tensor), The second input tensor of elementwise op.");
  }
  virtual void AddOpOutput() {
    AddOutput("Out",
              "N-dimension tensor. A location into which the result is stored. "
              "It's dimension "
              "equals with x");
  }
  virtual void AddOpComment() { AddComment(GetCommentExamples()); }

  virtual std::string GetOpFuntionality() const { return ""; }

  virtual std::string GetName() const = 0;
  virtual std::string GetEquation() const = 0;

  std::string GetCommentExamples() const {
    return string::Sprintf(R"DOC(
Elementwise %s Operator.

%s

The equation is:

$$%s$$

- $X$: a tensor of any dimension.
- $Y$: a tensor whose dimensions must be less than or equal to the dimensions of $X$.

There are two cases for this operator:

1. The shape of $Y$ is the same with $X$.
2. The shape of $Y$ is a continuous subsequence of $X$.

For case 2:

1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index
   for broadcasting $Y$ onto $X$.
2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of
   subsequence, such as shape(Y) = (2, 1) => (2).

For example:

  .. code-block:: text

    shape(X) = (2, 3, 4, 5), shape(Y) = (,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

)DOC",
                           GetName(), GetOpFuntionality(), GetEquation());
  }
};

class ElementwiseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(out_grad_name),
                   "Input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim(out_grad_name);
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_GE(
        x_dims.size(), y_dims.size(),
        "ShapeError: the dimension of Out@GRAD must greater than or equal to "
        "the one of input Y. But received: the shape of Out@GRAD = [%s], the "
        "dimension of Out@GRAD = %d, the shape of input Y = [%s], the "
        "dimension of of input Y = %d",
        x_dims, x_dims.size(), y_dims, y_dims.size());

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim(out_grad_name, /*->*/ x_grad_name);
      ctx->ShareLoD(out_grad_name, /*->*/ x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", /*->*/ y_grad_name);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type();

#ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ElementwiseOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", x_grad_name);
      ctx->ShareLoD("X", x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", y_grad_name);
      ctx->ShareLoD("Y", y_grad_name);
    }
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
      ctx->ShareLoD("DOut", "DDOut");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("DOut")->type();

#ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ElementwiseOpDoubleGradWithoutDXDY
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
      ctx->ShareLoD("DOut", "DDOut");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type input_data_type;
    if (ctx.HasInput("DDX") == false) {
      PADDLE_ENFORCE_EQ(ctx.HasInput("DDY"), true,
                        "Input(DDY) should not be null");
      input_data_type = ctx.Input<Tensor>("DDY")->type();
    } else if (ctx.HasInput("DDY") == false) {
      PADDLE_ENFORCE_EQ(ctx.HasInput("DDX"), true,
                        "Input(DDX) should not be null");
      input_data_type = ctx.Input<Tensor>("DDX")->type();
    } else {
      input_data_type = ctx.Input<Tensor>("DDX")->type();
    }

#ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

// For Add, Sub op, the X, Out is not needed.
class ElementwiseOpExplicitGrad : public ElementwiseOpGrad {
 public:
  using operators::ElementwiseOpGrad::ElementwiseOpGrad;
  using operators::ElementwiseOpGrad::GetExpectedKernelType;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim(framework::GradVarName("Out"), /*->*/ x_grad_name);
      ctx->ShareLoD(framework::GradVarName("Out"), /*->*/ x_grad_name);
    }
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(y_grad_name)) {
      PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");

      ctx->ShareDim("Y", /*->*/ y_grad_name);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);
    }
  }
};

template <typename T>
class ElemwiseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *dx =
        context.Output<framework::LoDTensor>(framework::GradVarName("X"));
    if (dx != nullptr) {
      auto &dout =
          *context.Input<framework::LoDTensor>(framework::GradVarName("Out"));
      dx->set_lod(dout.lod());
    }
  }
};

DECLARE_INPLACE_OP_INFERER(ElementwiseOpInplace, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(ElementwiseGradOpInplace,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_INPLACE_OP_INFERER(ElementwiseDoubleGradOpInplace, {"DDX", "DDOut"});

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ElementwiseGradNoBufVarsInference, "Y");
DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ElementwiseDoubleGradNoBufVarsInference,
                                      "Y", "DOut");

}  // namespace operators
}  // namespace paddle

#define REGISTER_ELEMWISE_GRAD_MAKER(kernel_type, op_name)                   \
  class kernel_type##GradMaker                                               \
      : public paddle::framework::SingleGradOpDescMaker {                    \
   public:                                                                   \
    using ::paddle::framework::SingleGradOpDescMaker::SingleGradOpDescMaker; \
                                                                             \
   protected:                                                                \
    std::unique_ptr<paddle::framework::OpDesc> Apply() const override {      \
      auto *op = new paddle::framework::OpDesc();                            \
      op->SetType(#kernel_type "_grad");                                     \
      op->SetInput("Y", Input("Y"));                                         \
      op->SetInput(::paddle::framework::GradVarName("Out"),                  \
                   OutputGrad("Out"));                                       \
      op->SetAttrMap(Attrs());                                               \
      op->SetOutput(::paddle::framework::GradVarName("X"), InputGrad("X"));  \
      op->SetOutput(::paddle::framework::GradVarName("Y"), InputGrad("Y"));  \
      return std::unique_ptr<::paddle::framework::OpDesc>(op);               \
    }                                                                        \
  }

#define REGISTER_ELEMWISE_OP(op_type, op_name, equation)                \
  class __ElemwiseOp##op_type##Maker__                                  \
      : public ::paddle::operators::ElementwiseOpMaker {                \
   protected:                                                           \
    virtual std::string GetName() const { return op_name; }             \
    virtual std::string GetEquation() const { return equation; }        \
  };                                                                    \
  REGISTER_OPERATOR(op_type, ::paddle::operators::ElementwiseOp,        \
                    __ElemwiseOp##op_type##Maker__,                     \
                    ::paddle::operators::ElementwiseOpInferVarType,     \
                    ::paddle::framework::DefaultGradOpDescMaker<true>); \
  REGISTER_OPERATOR(op_type##_grad, ::paddle::operators::ElementwiseOpGrad)

#define REGISTER_ELEMWISE_EXPLICIT_OP(op_type, op_name, equation)   \
  class __ElemwiseOp##op_type##Maker__                              \
      : public ::paddle::operators::ElementwiseOpMaker {            \
   protected:                                                       \
    virtual std::string GetName() const { return op_name; }         \
    virtual std::string GetEquation() const { return equation; }    \
  };                                                                \
  REGISTER_OPERATOR(op_type, ::paddle::operators::ElementwiseOp,    \
                    __ElemwiseOp##op_type##Maker__,                 \
                    ::paddle::operators::ElementwiseOpInferVarType, \
                    op_type##GradMaker,                             \
                    ::paddle::operators::ElementwiseOpInplace);     \
  REGISTER_OPERATOR(op_type##_grad,                                 \
                    ::paddle::operators::ElementwiseOpExplicitGrad, \
                    ::paddle::operators::ElementwiseGradOpInplace,  \
                    ::paddle::operators::ElementwiseGradNoBufVarsInference)

#define REGISTER_ELEMWISE_EXPLICIT_OP_WITHOUT_GRAD(op_type, op_name)    \
  REGISTER_OPERATOR(op_type, ::paddle::operators::ElementwiseOp,        \
                    ::paddle::operators::Elementwise##op_name##OpMaker, \
                    ::paddle::operators::ElementwiseOpInferVarType,     \
                    op_type##GradMaker,                                 \
                    ::paddle::operators::ElementwiseOpInplace);
