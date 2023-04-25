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

#include <algorithm>  // for max
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

class ElementwiseAddOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
#ifdef PADDLE_WITH_MKLDNN
      // When elementwise is first oneDNN op (there was some non oneDNN op
      // previously)
      // then we also need to rotate shape NHWC -> NCWH
      if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
          (tensor.layout() != phi::DataLayout::ONEDNN) &&
          phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
              phi::DataLayout::kNHWC) {
        return phi::KernelKey(tensor.place(),
                              phi::DataLayout::kNHWC,
                              expected_kernel_type.dtype());
      }
#endif
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class ElementwiseAddOpMaker : public framework::OpProtoAndCheckerMaker {
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
        .SetDefault(-1);
    AddOpComment();
  }

 protected:
  std::string GetName() const { return "Add"; }
  std::string GetEquation() const { return "Out = X + Y"; }

  void AddInputX() {
    AddInput(
        "X",
        "(Variable), Tensor or phi::DenseTensor of any dimensions. Its dtype "
        "should be int32, int64, float32, float64.");
  }

  void AddInputY() {
    AddInput(
        "Y",
        "(Variable), Tensor or phi::DenseTensor of any dimensions. Its dtype "
        "should be int32, int64, float32, float64.");
  }

  std::string GetOpFuntionality() const {
    return "Add two tensors element-wise";
  }
  void AddOpOutput() {
    AddOutput("Out",
              "N-dimension tensor. A location into which the result is stored. "
              "It's dimension "
              "equals with x");
  }
  void AddOpComment() { AddComment(GetCommentExamples()); }

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
                           GetName(),
                           GetOpFuntionality(),
                           GetEquation());
  }
};

class ElementwiseOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class ElementwiseAddCompositeGradOpMaker
    : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor y = this->GetSingleForwardInput("Y");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::Tensor dx = this->GetSingleInputGrad("X");
    auto *dx_ptr = this->GetOutputPtr(&dx);
    std::string dx_name = this->GetOutputName(dx);
    paddle::Tensor dy = this->GetSingleInputGrad("Y");
    auto *dy_ptr = this->GetOutputPtr(&dy);
    std::string dy_name = this->GetOutputName(dy);
    int axis = static_cast<int>(this->Attr<int>("axis"));
    PADDLE_ENFORCE_EQ(
        axis,
        -1,
        phi::errors::InvalidArgument(
            "We only support axis = -1 in composite add_grad but we got: ",
            axis));
    VLOG(6) << "Runing add_grad composite func";
    prim::add_grad<prim::DescTensor>(x, y, out_grad, axis, dx_ptr, dy_ptr);
    this->RecoverOutputName(dx, dx_name);
    this->RecoverOutputName(dy, dy_name);
  }
};

template <typename T>
class ElementwiseAddDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_add_grad_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

class ElementwiseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ElementwiseOpGrad");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name),
                   "Input",
                   out_grad_name,
                   "ElementwiseOpGrad");
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", /*->*/ x_grad_name);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", /*->*/ y_grad_name);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class ElementwiseOpDoubleGradWithoutDXDY
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
      ctx->ShareLoD("DOut", "DDOut");
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type input_data_type;
    if (ctx.HasInput("DDX") == false) {
      OP_INOUT_CHECK(ctx.HasInput("DDY"),
                     "Input",
                     "DDY",
                     "ElementwiseOpDoubleGradWithoutDXDY");
      input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DDY");
    } else if (ctx.HasInput("DDY") == false) {
      OP_INOUT_CHECK(ctx.HasInput("DDX"),
                     "Input",
                     "DDX",
                     "ElementwiseOpDoubleGradWithoutDXDY");
      input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DDX");
    } else {
      input_data_type =
          OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "DDX", "DDY");
    }
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class ElementwiseOpTripleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasOutput("D_DDX")) {
      ctx->ShareDim("DDX", "D_DDX");
      ctx->ShareLoD("DDX", "D_DDX");
    }
    if (ctx->HasOutput("D_DDY")) {
      ctx->ShareDim("DDY", "D_DDY");
      ctx->ShareLoD("DDY", "D_DDY");
    }
    if (ctx->HasOutput("D_X")) {
      ctx->ShareDim("X", "D_X");
      ctx->ShareLoD("X", "D_X");
    }
    if (ctx->HasOutput("D_Y")) {
      ctx->ShareDim("Y", "D_Y");
      ctx->ShareLoD("Y", "D_Y");
    }
    if (ctx->HasOutput("D_DOut")) {
      ctx->ShareDim("DOut", "D_DOut");
      ctx->ShareLoD("DOut", "D_DOut");
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type input_data_type;
    input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "D_DDOut");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

template <typename T>
class ElementwiseAddTripleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_add_triple_grad");
    op->SetInput("DDX", this->Input("DDX"));
    op->SetInput("DDY", this->Input("DDY"));
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("D_DDX", this->InputGrad("DDX"));
    op->SetOutput("D_DDY", this->InputGrad("DDY"));
  }
};

DECLARE_INFER_SHAPE_FUNCTOR(elementwise_add,
                            ElementwiseAddInferShapeFunctor,
                            PD_INFER_META(phi::ElementwiseRawInferMeta));

DECLARE_INPLACE_OP_INFERER(ElementwiseOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(ElementwiseGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_INPLACE_OP_INFERER(ElementwiseDoubleGradOpInplaceInferer,
                           {"DDX", "DDOut"});

DECLARE_INPLACE_OP_INFERER(ElementwiseTripleGradOpInplaceInferer,
                           {"D_DDOut", "D_DDX"});

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseGradNoBufVarsInferer, "X", "Y");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseDoubleGradNoBufVarsInferer,
                                    "Y",
                                    "DOut");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseTripleGradNoBufVarsInferer,
                                    "DDX",
                                    "DDY");

}  // namespace operators
}  // namespace paddle

#define REGISTER_ELEMWISE_GRAD_MAKER(kernel_type, op_name)              \
  template <typename T>                                                 \
  class kernel_type##GradMaker                                          \
      : public paddle::framework::SingleGradOpMaker<T> {                \
   public:                                                              \
    using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker; \
                                                                        \
   protected:                                                           \
    void Apply(::paddle::framework::GradOpPtr<T> op) const override {   \
      op->SetType(#kernel_type "_grad");                                \
      op->SetInput("X", this->Input("X"));                              \
      op->SetInput("Y", this->Input("Y"));                              \
      op->SetInput(::paddle::framework::GradVarName("Out"),             \
                   this->OutputGrad("Out"));                            \
      op->SetAttrMap(this->Attrs());                                    \
      op->SetOutput(::paddle::framework::GradVarName("X"),              \
                    this->InputGrad("X"));                              \
      op->SetOutput(::paddle::framework::GradVarName("Y"),              \
                    this->InputGrad("Y"));                              \
    }                                                                   \
  }

#define REGISTER_ELEMWISE_GRAD_MAKER(kernel_type, op_name)              \
  template <typename T>                                                 \
  class kernel_type##GradMaker                                          \
      : public paddle::framework::SingleGradOpMaker<T> {                \
   public:                                                              \
    using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker; \
                                                                        \
   protected:                                                           \
    void Apply(::paddle::framework::GradOpPtr<T> op) const override {   \
      op->SetType(#kernel_type "_grad");                                \
      op->SetInput("X", this->Input("X"));                              \
      op->SetInput("Y", this->Input("Y"));                              \
      op->SetInput(::paddle::framework::GradVarName("Out"),             \
                   this->OutputGrad("Out"));                            \
      op->SetAttrMap(this->Attrs());                                    \
      op->SetOutput(::paddle::framework::GradVarName("X"),              \
                    this->InputGrad("X"));                              \
      op->SetOutput(::paddle::framework::GradVarName("Y"),              \
                    this->InputGrad("Y"));                              \
    }                                                                   \
  }

REGISTER_ELEMWISE_GRAD_MAKER(elementwise_add, Add);
REGISTER_OPERATOR(elementwise_add,
                  ::paddle::operators::ElementwiseAddOp,
                  ::paddle::operators::ElementwiseAddOpMaker,
                  ::paddle::operators::ElementwiseOpInferVarType,
                  elementwise_addGradMaker<::paddle::framework::OpDesc>,
                  elementwise_addGradMaker<::paddle::imperative::OpBase>,
                  ::paddle::operators::ElementwiseAddCompositeGradOpMaker,
                  ::paddle::operators::ElementwiseAddInferShapeFunctor);

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    elementwise_add_grad,
    ops::ElementwiseOpGrad,
    ops::ElementwiseGradOpInplaceInferer,
    ops::ElementwiseGradNoBufVarsInferer,
    ops::ElementwiseAddDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseAddDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    elementwise_add_grad_grad,
    ops::ElementwiseOpDoubleGradWithoutDXDY,
    ops::ElementwiseDoubleGradOpInplaceInferer,
    ops::ElementwiseDoubleGradNoBufVarsInferer,
    ops::ElementwiseAddTripleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseAddTripleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_add_triple_grad,
                  ops::ElementwiseOpTripleGrad,
                  ops::ElementwiseTripleGradOpInplaceInferer,
                  ops::ElementwiseTripleGradNoBufVarsInferer);

// A specialization elementwise_add operator, used in gradient accumulation with
// inplace addto.
REGISTER_OPERATOR(
    grad_add,
    paddle::operators::ElementwiseAddOp,
    paddle::operators::ElementwiseAddOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_VERSION(elementwise_add)
    .AddCheckpoint(
        R"ROC(Register elementwise_add for adding the attribute of
       Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_add.",
            1.0f));
