//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/matmul_v2_op.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"

namespace paddle {
namespace operators {

void MatMulV2OpMaker::Make() {
  AddInput("X", "tensor of shape (d0, d1 ... M, K)");
  AddInput("Y", "tensor of shape (d0, d1 ... K, N)");
  AddOutput("Out", "tensor of shape (d0, d1 ... M, N)");
  AddAttr<bool>("trans_x",
                "Set true to transpose the last two dimensions of X before "
                "doing multiplication")
      .SetDefault(false);
  AddAttr<bool>("trans_y",
                "Set true to transpose the last two dimensions of Y before "
                "doing multiplication")
      .SetDefault(false);
  AddComment(
      R"DOC(Matrix multiplication Out = X * Y. A has shape (d0, d1 ... M, K),
        B has shape (d0, d1 ... K, N), Out has shape ((d0, d1 ... M, N)).
        In addition, it also follows the broadcast rule which is similar as
        numpy.matmul.
)DOC");
  Apply();
}

class MatMulV2OpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
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
class MatMulV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("matmul_v2_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

class MatMulV2OpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput("DOut"), "Input", "DOut", "matmul");

    if (context->HasOutput("DX") && context->HasInput("DDY")) {
      context->ShareDim("X", "DX");
    }

    if (context->HasOutput("DY") && context->HasInput("DDX")) {
      context->ShareDim("Y", "DY");
    }

    if (context->HasOutput("DDOut") &&
        (context->HasInput("DDY") || context->HasInput("DDX"))) {
      context->ShareDim("DOut", "DDOut");
    }
  }
};

template <typename T>
class MatMulV2OpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("matmul_v2_grad_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    auto ddx = this->OutputGrad(framework::GradVarName("X"));
    auto ddy = this->OutputGrad(framework::GradVarName("Y"));

    if (!ddx.empty() || !ddy.empty()) {
      op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    }
    op->SetOutput("DX",
                  ddy.empty() ? this->EmptyInputGrad() : this->InputGrad("X"));
    op->SetOutput("DY",
                  ddx.empty() ? this->EmptyInputGrad() : this->InputGrad("Y"));

    op->SetAttrMap(this->Attrs());
  }
};
class MatMulV2OpTripleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(
        context->HasInput("X"), "Input", "X", "matmul_v2_triple_grad");
    OP_INOUT_CHECK(
        context->HasInput("Y"), "Input", "Y", "matmul_v2_triple_grad");
    OP_INOUT_CHECK(
        context->HasInput("DOut"), "Input", "DOut", "matmul_v2_triple_grad");
    OP_INOUT_CHECK(
        context->HasInput("DDX"), "Input", "DDX", "matmul_v2_triple_grad");
    OP_INOUT_CHECK(
        context->HasInput("DDY"), "Input", "DDY", "matmul_v2_triple_grad");
    OP_INOUT_CHECK(
        context->HasInput("D_DX"), "Input", "D_DX", "matmul_v2_triple_grad");
    OP_INOUT_CHECK(
        context->HasInput("D_DY"), "Input", "D_DY", "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("D_DDOut"),
                   "Input",
                   "D_DDOut",
                   "matmul_v2_triple_grad");

    if (context->HasOutput("D_X_out")) {
      context->ShareDim("X", "D_X_out");
    }
    if (context->HasOutput("D_Y_out")) {
      context->ShareDim("Y", "D_Y_out");
    }
    if (context->HasOutput("D_DOut_out")) {
      context->ShareDim("DOut", "D_DOut_out");
    }
    if (context->HasOutput("D_DDX_out")) {
      context->ShareDim("X", "D_DDX_out");
    }
    if (context->HasOutput("D_DDY_out")) {
      context->ShareDim("Y", "D_DDY_out");
    }
  }
};

template <typename T>
class MatMulV2OpTripleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("matmul_v2_triple_grad");

    // get input from double grad
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input("DOut"));
    op->SetInput("DDX", this->Input("DDX"));
    op->SetInput("DDY", this->Input("DDY"));
    op->SetInput("D_DX", this->OutputGrad("DX"));
    op->SetInput("D_DY", this->OutputGrad("DY"));
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));

    // set outputs
    op->SetOutput("D_X_out", this->InputGrad("X"));
    op->SetOutput("D_Y_out", this->InputGrad("Y"));
    op->SetOutput("D_DOut_out", this->InputGrad("DOut"));
    op->SetOutput("D_DDX_out", this->InputGrad("DDX"));
    op->SetOutput("D_DDY_out", this->InputGrad("DDY"));

    op->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(matmul_v2,
                  ops::MatMulV2Op,
                  ops::MatMulV2OpMaker,
                  ops::MatMulV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::MatMulV2GradOpMaker<paddle::imperative::OpBase>);

DECLARE_INFER_SHAPE_FUNCTOR(matmul_v2_grad,
                            MatMulV2GradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralBinaryGradInferMeta));
REGISTER_OPERATOR(matmul_v2_grad,
                  ops::MatMulV2OpGrad,
                  ops::MatMulV2OpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulV2OpDoubleGradMaker<paddle::imperative::OpBase>,
                  MatMulV2GradInferShapeFunctor);

REGISTER_OPERATOR(matmul_v2_grad_grad,
                  ops::MatMulV2OpDoubleGrad,
                  ops::MatMulV2OpTripleGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulV2OpTripleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(matmul_v2_triple_grad, ops::MatMulV2OpTripleGrad);
