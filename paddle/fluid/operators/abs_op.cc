// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class AbsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class AbsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of abs op.");
    AddOutput("Out", "(Tensor), The output tensor of abs op.");
    AddComment(R"DOC(
Abs Operator.

This operator is used to perform elementwise abs for input $X$.
$$out = |x|$$

)DOC");
  }
};

class AbsGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@Grad",
                   "AbsGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   "X@Grad",
                   "AbsGrad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class AbsGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("abs_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

// AbsGrad: dx=dy if x >=0 else -dy
// AbsDoubleGrad: ddy = ddx if x >=0 else -ddx
template <typename T>
class AbsDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("abs_double_grad");
    // input1: x
    op->SetInput("X", this->Input("X"));
    // input2: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());
    // output: ddy
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

class AbsDoubleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("X", "DDOut");
      ctx->ShareLoD("X", "DDOut");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "DDX");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(
        framework::TransToProtoVarType(tensor.dtype()),
        tensor.place(),
        tensor.layout());
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(abs,
                            AbsInferShapeFunctor,
                            PD_INFER_META(phi::RealAndImagInferMeta));

namespace ops = paddle::operators;

REGISTER_OPERATOR(abs,
                  ops::AbsOp,
                  ops::AbsOpMaker,
                  ops::AbsGradMaker<paddle::framework::OpDesc>,
                  ops::AbsGradMaker<paddle::imperative::OpBase>,
                  AbsInferShapeFunctor);

REGISTER_OPERATOR(abs_grad,
                  ops::AbsGradOp,
                  ops::AbsDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::AbsDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(abs_double_grad, ops::AbsDoubleGradOp);
