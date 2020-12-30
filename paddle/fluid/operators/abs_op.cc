/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/abs_op.h"

#include <string>

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename GradFunctor>
static constexpr bool CanInplaceAct() {
  return GradFunctor::FwdDeps() == kDepOut || GradFunctor::FwdDeps() == kNoDeps;
}

framework::OpKernelType GetAbsKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel& oper, const std::string& name) {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
#ifdef PADDLE_WITH_MKLDNN
  auto it = oper.Attrs().find("use_mkldnn");
  if (library == framework::LibraryType::kPlain && it != oper.Attrs().end() &&
      oper.CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif
  return framework::OpKernelType(oper.IndicateVarDataType(ctx, name),
                                 ctx.GetPlace(), layout, library);
}

class AbsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetAbsKernelType(ctx, *this, "X");
  }
};

class AbsOpMaker : public ::paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Abs operator");
    AddOutput("Out", "Output of Abs operator");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<bool>("use_cudnn",
                  "(bool, default false) Only used in cudnn kernel, need "
                  "install cudnn")
        .SetDefault(false);
    AddComment(R"DOC(
Abs Operator.

$$out = |x|$$

)DOC");
  }
};

class AbsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    ctx->ShareDim(out_grad_name, framework::GradVarName("X"));
    ctx->ShareLoD(out_grad_name, framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetAbsKernelType(ctx, *this, "X");
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(tensor.type(), tensor.place(),
                                   tensor.layout());
  }
};

template <ActBwdOpFwdDeps kDepValue, typename T>
class AbsGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());

    if ((static_cast<int>(kDepValue) &
         static_cast<int>(ActBwdOpFwdDeps::kDepX)) ||
        FLAGS_use_mkldnn ||
        (op->HasAttr("use_mkldnn") &&
         BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")))) {
      op->SetInput("X", this->Input("X"));
    }

    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      op->SetInput("Out", this->Output("Out"));
    }
  }
};

template <ActBwdOpFwdDeps kDepValue>
class AbsOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (static_cast<int>(kDepValue) & static_cast<int>(kDepX)) {
      if (ctx->HasOutput("DX")) {
        ctx->ShareDim("X", "DX");
        ctx->ShareLoD("X", "DX");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("X", "DDOut");
        ctx->ShareLoD("X", "DDOut");
      }
    }
    if (static_cast<int>(kDepValue) & static_cast<int>(kDepOut)) {
      if (ctx->HasOutput("DOut")) {
        ctx->ShareDim("Out", "DOut");
        ctx->ShareLoD("Out", "DOut");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("Out", "DDOut");
        ctx->ShareLoD("Out", "DDOut");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetAbsKernelType(ctx, *this, "DDX");
  }
};

// AbsGrad: dx=dy if x >=0 else -dy
// AbsDoubleGrad: ddy = ddx if x >=0 else -ddx
template <typename T>
class AbsDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("abs_grad_grad");
    // input1: x
    op->SetInput("X", this->Input("X"));
    // input2: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());
    // output: ddy
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// DECLARE_INPLACE_OP_INFERER(AbsInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(AbsGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_INPLACE_OP_INFERER(AbsDoubleGradOpInplaceInferer, {"DDX", "DDOut"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

/* ==========================   abs register  ============================ */
REGISTER_OPERATOR(abs, ops::AbsOp, ops::AbsOpMaker,
                  ops::AbsGradOpMaker<ops::AbsGradFunctor<float>::FwdDeps(),
                                      paddle::framework::OpDesc>,
                  ops::AbsGradOpMaker<ops::AbsGradFunctor<float>::FwdDeps(),
                                      paddle::imperative::OpBase>);
REGISTER_OPERATOR(abs_grad, ops::AbsOpGrad, ops::AbsGradOpInplaceInferer,
                  ops::AbsDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::AbsDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    abs_grad_grad,
    ops::AbsOpDoubleGrad<ops::AbsGradGradFunctor<float>::FwdDeps()>,
    ops::AbsDoubleGradOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    abs,
    ops::AbsKernel<paddle::platform::CPUDeviceContext, ops::AbsFunctor<float>>,
    ops::AbsKernel<paddle::platform::CPUDeviceContext, ops::AbsFunctor<double>>,
    ops::AbsKernel<paddle::platform::CPUDeviceContext, ops::AbsFunctor<int>>,
    ops::AbsKernel<paddle::platform::CPUDeviceContext,
                   ops::AbsFunctor<int64_t>>,
    ops::AbsKernel<paddle::platform::CPUDeviceContext,
                   ops::AbsFunctor<plat::complex64>>,
    ops::AbsKernel<paddle::platform::CPUDeviceContext,
                   ops::AbsFunctor<plat::complex128>>);
REGISTER_OP_CPU_KERNEL(
    abs_grad, ops::AbsGradKernel<paddle::platform::CPUDeviceContext,
                                 ops::AbsGradFunctor<float>>,
    ops::AbsGradKernel<paddle::platform::CPUDeviceContext,
                       ops::AbsGradFunctor<double>>,
    ops::AbsGradKernel<paddle::platform::CPUDeviceContext,
                       ops::AbsGradFunctor<int>>,
    ops::AbsGradKernel<paddle::platform::CPUDeviceContext,
                       ops::AbsGradFunctor<int64_t>>,
    ops::AbsGradKernel<paddle::platform::CPUDeviceContext,
                       ops::AbsGradFunctor<plat::complex64>>,
    ops::AbsGradKernel<paddle::platform::CPUDeviceContext,
                       ops::AbsGradFunctor<plat::complex128>>);
REGISTER_OP_CPU_KERNEL(
    abs_grad_grad, ops::AbsDoubleGradKernel<plat::CPUDeviceContext,
                                            ops::AbsGradGradFunctor<float>>,
    ops::AbsDoubleGradKernel<plat::CPUDeviceContext,
                             ops::AbsGradGradFunctor<double>>,
    ops::AbsDoubleGradKernel<plat::CPUDeviceContext,
                             ops::AbsGradGradFunctor<plat::float16>>,
    ops::AbsDoubleGradKernel<plat::CPUDeviceContext,
                             ops::AbsGradGradFunctor<int>>,
    ops::AbsDoubleGradKernel<plat::CPUDeviceContext,
                             ops::AbsGradGradFunctor<int64_t>>);
/* ========================================================================== */
