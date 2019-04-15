/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/activation_op.h"
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/operators/mkldnn/mkldnn_activation_op.h"
#include "paddle/fluid/platform/port.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename GradFunctor>
static constexpr bool CanInplaceAct() {
  return GradFunctor::FwdDeps() == kDepOut || GradFunctor::FwdDeps() == kNoDeps;
}

std::unique_ptr<std::unordered_set<std::string>> GetInplaceOpSet() {
  std::unique_ptr<std::unordered_set<std::string>> ret(
      new std::unordered_set<std::string>());
#define INSERT_INTO_INPLACE_OP_SET(op_type, __omitted, fwd_functor, \
                                   bwd_functor)                     \
  if (CanInplaceAct<bwd_functor<float>>()) {                        \
    ret->insert(#op_type);                                          \
  }

  FOR_EACH_ACTIVATION_OP(INSERT_INTO_INPLACE_OP_SET);
#undef INSERT_INTO_INPLACE_OP_SET
  return ret;
}

#define REGISTER_ACTIVATION_OP_MAKER(OP_NAME, OP_COMMENT)                    \
  class OP_NAME##OpMaker                                                     \
      : public ::paddle::framework::OpProtoAndCheckerMaker {                 \
   public:                                                                   \
    void Make() override {                                                   \
      AddInput("X", "Input of " #OP_NAME " operator");                       \
      AddOutput("Out", "Output of " #OP_NAME " operator");                   \
      AddAttr<bool>("use_mkldnn",                                            \
                    "(bool, default false) Only used in mkldnn kernel")      \
          .SetDefault(false);                                                \
      AddAttr<bool>("use_cudnn",                                             \
                    "(bool, default false) Only used in cudnn kernel, need " \
                    "install cudnn")                                         \
          .SetDefault(false);                                                \
      AddAttr<bool>(                                                         \
          "is_test",                                                         \
          "(bool, default false) Set to true for inference only, false "     \
          "for training. Some layers may run faster when this is true.")     \
          .SetDefault(false);                                                \
      AddComment(OP_COMMENT);                                                \
    }                                                                        \
  }

template <ActBwdOpFwdDeps kDepValue>
class ActivationGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType(ForwardOpType() + "_grad");
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());

    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
      op->SetInput("X", Input("X"));
    }

    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      op->SetInput("Out", Output("Out"));
    }

    return op;
  }
};

framework::OpKernelType GetKernelType(const framework::ExecutionContext& ctx,
                                      const framework::OperatorWithKernel& oper,
                                      const std::string& name) {
  framework::LibraryType library{framework::LibraryType::kPlain};
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
// FIXME(liuwei1031) temporarily disable the code to unblock users
// TODO(liuwei1031) figure out the reason behind
// https://github.com/PaddlePaddle/Paddle/issues/16096
// and re-enable this in the future
// #ifdef PADDLE_WITH_CUDA
//   auto it1 = oper.Attrs().find("use_cudnn");
//   if (it1 != oper.Attrs().end() && platform::CanCUDNNBeUsed(ctx)) {
//     library = framework::LibraryType::kCUDNN;
//   }
// #endif
#ifdef PADDLE_WITH_MKLDNN
  auto it = oper.Attrs().find("use_mkldnn");
  if (library == framework::LibraryType::kPlain && it != oper.Attrs().end() &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif
  return framework::OpKernelType(
      framework::GetDataTypeOfVar(ctx.InputVar(name)), ctx.GetPlace(), layout,
      library);
}

class ActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "X");
  }
};

class ActivationOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};

class ActivationOpGrad : public framework::OperatorWithKernel {
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
    return GetKernelType(ctx, *this, framework::GradVarName("Out"));
  }
};

UNUSED constexpr char SigmoidDoc[] = R"DOC(
Sigmoid Activation Operator

$$out = \\frac{1}{1 + e^{-x}}$$

)DOC";

UNUSED constexpr char LogSigmoidDoc[] = R"DOC(
Logsigmoid Activation Operator

$$out = \\log \\frac{1}{1 + e^{-x}}$$

)DOC";

UNUSED constexpr char ExpDoc[] = R"DOC(
Exp Activation Operator.

$out = e^x$

)DOC";

UNUSED constexpr char ReluDoc[] = R"DOC(
Relu Activation Operator.

$out = \max(x, 0)$

)DOC";

UNUSED constexpr char GeluDoc[] = R"DOC(
Gelu Activation Operator.

$out = \\frac{1 + erf(\\frac{x}{\\sqrt{2}})}{2} x$

)DOC";

UNUSED constexpr char TanhDoc[] = R"DOC(
Tanh Activation Operator.

$$out = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

UNUSED constexpr char TanhShrinkDoc[] = R"DOC(
TanhShrink Activation Operator.

$$out = x - \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

UNUSED constexpr char SqrtDoc[] = R"DOC(
Sqrt Activation Operator.

Please make sure legal input, when input a negative value closed to zero,
you should add a small epsilon(1e-12) to avoid negative number caused by numerical errors.

$out = \sqrt{x}$

)DOC";

UNUSED constexpr char AbsDoc[] = R"DOC(
Abs Activation Operator.

$out = |x|$

)DOC";

UNUSED constexpr char CeilDoc[] = R"DOC(
Ceil Activation Operator.

$out = \left \lceil x \right \rceil$

)DOC";

UNUSED constexpr char FloorDoc[] = R"DOC(
Floor Activation Operator.

$out = \left \lfloor x \right \rfloor$

)DOC";

UNUSED constexpr char CosDoc[] = R"DOC(
Cosine Activation Operator.

$out = cos(x)$

)DOC";

UNUSED constexpr char SinDoc[] = R"DOC(
Sine Activation Operator.

$out = sin(x)$

)DOC";

UNUSED constexpr char RoundDoc[] = R"DOC(
Round Activation Operator.

$out = [x]$

)DOC";

UNUSED constexpr char ReciprocalDoc[] = R"DOC(
Reciprocal Activation Operator.

$$out = \\frac{1}{x}$$

)DOC";

UNUSED constexpr char LogDoc[] = R"DOC(
Log Activation Operator.

$out = \ln(x)$

Natural logarithm of x.

)DOC";

UNUSED constexpr char SquareDoc[] = R"DOC(
Square Activation Operator.

$out = x^2$

)DOC";

UNUSED constexpr char SoftplusDoc[] = R"DOC(
Softplus Activation Operator.

$out = \ln(1 + e^{x})$

)DOC";

UNUSED constexpr char SoftsignDoc[] = R"DOC(
Softsign Activation Operator.

$$out = \\frac{x}{1 + \|x\|}$$

)DOC";

class AcosOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of acos operator");
    AddOutput("Out", "Output of acos operator");
    AddComment(R"DOC(
Arccosine Activation Operator.

$$out = \cos^{-1}(x)$$

)DOC");
  }
};

class AsinOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of asin operator");
    AddOutput("Out", "Output of asin operator");
    AddComment(R"DOC(
Arcsine Activation Operator.

$$out = \sin^{-1}(x)$$

)DOC");
  }
};

class AtanOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of atan operator");
    AddOutput("Out", "Output of atan operator");
    AddComment(R"DOC(
Arctanh Activation Operator.

$$out = \tanh^{-1}(x)$$

)DOC");
  }
};

class LeakyReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of LeakyRelu operator");
    AddOutput("Out", "Output of LeakyRelu operator");
    AddAttr<float>("alpha", "The small negative slope").SetDefault(0.02f);
    AddComment(R"DOC(
LeakyRelu Activation Operator.

$out = \max(x, \alpha * x)$

)DOC");
  }
};

class SoftShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Softshrink operator");
    AddOutput("Out", "Output of Softshrink operator");
    AddAttr<float>("lambda", "non-negative offset").SetDefault(0.5f);
    AddComment(R"DOC(
:strong:`Softshrink Activation Operator`

..  math::
    out = \begin{cases}
         x - \lambda, \text{if } x > \lambda \\
         x + \lambda, \text{if } x < -\lambda \\
         0,  \text{otherwise}
         \end{cases}

)DOC");
  }
};

class HardShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of HardShrink operator");
    AddOutput("Out", "Output of HardShrink operator");
    AddAttr<float>("threshold",
                   "The value of threshold for HardShrink. [default: 0.5]")
        .SetDefault(0.5f);
    AddComment(R"DOC(
:strong:`HardShrink activation operator`

..  math::
    out = \begin{cases}
            x, \text{if } x > \lambda \\
            x, \text{if } x < -\lambda \\
            0,  \text{otherwise}
          \end{cases}

)DOC");
  }
};

class BReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of BRelu operator");
    AddOutput("Out", "Output of BRelu operator");
    AddAttr<float>("t_min", "The min marginal value of BRelu")
        .SetDefault(static_cast<float>(0));
    AddAttr<float>("t_max", "The max marginal value of BRelu")
        .SetDefault(static_cast<float>(24));
    AddComment(R"DOC(
BRelu Activation Operator.

$out = \max(\min(x, t_{min}), t_{max})$

)DOC");
  }
};

class SoftReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of SoftRelu operator");
    AddOutput("Out", "Output of SoftRelu operator");
    AddAttr<float>("threshold", "The threshold value of SoftRelu")
        .SetDefault(40.0f);
    AddComment(R"DOC(
SoftRelu Activation Operator.

$out = \ln(1 + \exp(\max(\min(x, threshold), threshold))$

)DOC");
  }
};

class ELUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of ELU operator");
    AddOutput("Out", "Output of ELU operator");
    AddAttr<float>("alpha", "The alpha value of ELU").SetDefault(1.0f);
    AddComment(R"DOC(
ELU Activation Operator.

Applies the following element-wise computation on the input according to
https://arxiv.org/abs/1511.07289.

$out = \max(0, x) + \min(0, \alpha * (e^x - 1))$

)DOC");
  }
};

class Relu6OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Relu6 operator");
    AddOutput("Out", "Output of Relu6 operator");
    AddAttr<float>("threshold", "The threshold value of Relu6")
        .SetDefault(6.0f);
    AddComment(R"DOC(
Relu6 Activation Operator.

$out = \min(\max(0, x), 6)$

)DOC");
  }
};

class PowOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Pow operator");
    AddOutput("Out", "Output of Pow operator");
    AddAttr<float>("factor", "The exponential factor of Pow").SetDefault(1.0f);
    AddComment(R"DOC(
Pow Activation Operator.

$out = x^{factor}$

)DOC");
  }
};

class STanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of STanh operator");
    AddOutput("Out", "Output of STanh operator");
    AddAttr<float>("scale_a", "The scale parameter of a for the input")
        .SetDefault(2.0f / 3.0f);
    AddAttr<float>("scale_b", "The scale parameter of b for the input")
        .SetDefault(1.7159f);
    AddComment(R"DOC(
STanh Activation Operator.

$$out = b * \\frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}$$

)DOC");
  }
};

class ThresholdedReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of ThresholdedRelu operator");
    AddOutput("Out", "Output of ThresholdedRelu operator");
    AddAttr<float>("threshold",
                   "The threshold location of activation. [default 1.0].")
        .SetDefault(1.0f);
    AddComment(R"DOC(
:strong:`ThresholdedRelu activation operator`

..  math::

    out = \begin{cases}
             x,  \text{if } x > threshold \\
             0,  \text{otherwise}
          \end{cases}
)DOC");
  }
};

class HardSigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of HardSigmoid operator");
    AddOutput("Out", "Output of HardSigmoid operator");
    AddAttr<float>("slope", "Slope for linear approximation of sigmoid")
        .SetDefault(0.2f);
    AddAttr<float>("offset", "Offset for linear approximation of sigmoid")
        .SetDefault(0.5f);
    AddComment(R"DOC(
HardSigmoid Activation Operator.

Segment-wise linear approximation of sigmoid(https://arxiv.org/abs/1603.00391),
which is much faster than sigmoid.

$out = \max(0, \min(1, slope * x + shift))$

The slope should be positive. The offset can be either positive or negative.
The default slope and shift are set according to the above reference.
It is recommended to use the defaults for this activation.

)DOC");
  }
};

class SwishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Swish operator");
    AddOutput("Out", "Output of Swish operator");
    AddAttr<float>("beta", "Constant beta of swish operator").SetDefault(1.0f);
    AddComment(R"DOC(
Swish Activation Operator.

$$out = \\frac{x}{1 + e^{- \beta x}}$$

)DOC");
  }
};

REGISTER_ACTIVATION_OP_MAKER(Sigmoid, SigmoidDoc);
REGISTER_ACTIVATION_OP_MAKER(LogSigmoid, LogSigmoidDoc);
REGISTER_ACTIVATION_OP_MAKER(Exp, ExpDoc);
REGISTER_ACTIVATION_OP_MAKER(Relu, ReluDoc);
REGISTER_ACTIVATION_OP_MAKER(Gelu, GeluDoc);
REGISTER_ACTIVATION_OP_MAKER(Tanh, TanhDoc);
REGISTER_ACTIVATION_OP_MAKER(TanhShrink, TanhShrinkDoc);
REGISTER_ACTIVATION_OP_MAKER(Sqrt, SqrtDoc);
REGISTER_ACTIVATION_OP_MAKER(Abs, AbsDoc);
REGISTER_ACTIVATION_OP_MAKER(Ceil, CeilDoc);
REGISTER_ACTIVATION_OP_MAKER(Floor, FloorDoc);
REGISTER_ACTIVATION_OP_MAKER(Cos, CosDoc);
REGISTER_ACTIVATION_OP_MAKER(Sin, SinDoc);
REGISTER_ACTIVATION_OP_MAKER(Round, RoundDoc);
REGISTER_ACTIVATION_OP_MAKER(Reciprocal, ReciprocalDoc);
REGISTER_ACTIVATION_OP_MAKER(Log, LogDoc);
REGISTER_ACTIVATION_OP_MAKER(Square, SquareDoc);
REGISTER_ACTIVATION_OP_MAKER(Softplus, SoftplusDoc);
REGISTER_ACTIVATION_OP_MAKER(Softsign, SoftsignDoc);

class ActivationOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("Out"))) {
      ctx->ShareDim("Out", framework::GradVarName("Out"));
      ctx->ShareLoD("Out", framework::GradVarName("Out"));
    }
    if (ctx->HasOutput(framework::DoubleGradVarName("Out"))) {
      ctx->ShareDim("Out", framework::DoubleGradVarName("Out"));
      ctx->ShareLoD("Out", framework::DoubleGradVarName("Out"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "Out");
  }
};

//
// ReluGrad: dx = dy if y >= 0 else 0
// ReluGradGrad: ddy = ddx if y >= 0 else 0
//
class ReluDoubleGradMaker : public ::paddle::framework::SingleGradOpDescMaker {
 public:
  using ::paddle::framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<::paddle::framework::OpDesc> Apply() const override {
    auto* op = new ::paddle::framework::OpDesc();
    op->SetType("relu_grad_grad");
    // input1: Out
    op->SetInput("Out", Input("Out"));
    // X@GRAD@GRAD: ddx
    op->SetInput(framework::DoubleGradVarName("X"),
                 OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(Attrs());
    // Out@GRAD@GRAD: ddy
    op->SetOutput(framework::GradVarName("Out"), InputGrad("Out"));
    op->SetOutput(framework::DoubleGradVarName("Out"),
                  InputGrad(framework::GradVarName("Out")));
    return std::unique_ptr<::paddle::framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_OP(KERNEL_TYPE, OP_NAME, functor, grad_functor) \
  REGISTER_OPERATOR(                                                        \
      KERNEL_TYPE, ops::ActivationOp, ops::OP_NAME##OpMaker,                \
      ops::ActivationOpInferVarType,                                        \
      ops::ActivationGradOpDescMaker<ops::grad_functor<float>::FwdDeps()>,  \
      std::conditional<ops::CanInplaceAct<ops::grad_functor<float>>(),      \
                       ::paddle::framework::SingleOpInplaceInToOut,         \
                       void>::type);                                        \
  REGISTER_OPERATOR(                                                        \
      KERNEL_TYPE##_grad, ops::ActivationOpGrad,                            \
      std::conditional<ops::CanInplaceAct<ops::grad_functor<float>>(),      \
                       ::paddle::framework::SingleOpInplaceInToOut,         \
                       void>::type)

#define REGISTER_ACTIVATION_CPU_KERNEL(act_type, op_name, functor,        \
                                       grad_functor)                      \
  REGISTER_OP_CPU_KERNEL(                                                 \
      act_type, ops::ActivationKernel<paddle::platform::CPUDeviceContext, \
                                      ops::functor<float>>,               \
      ops::ActivationKernel<paddle::platform::CPUDeviceContext,           \
                            ops::functor<double>>);                       \
  REGISTER_OP_CPU_KERNEL(                                                 \
      act_type##_grad,                                                    \
      ops::ActivationGradKernel<paddle::platform::CPUDeviceContext,       \
                                ops::grad_functor<float>>,                \
      ops::ActivationGradKernel<paddle::platform::CPUDeviceContext,       \
                                ops::grad_functor<double>>);

FOR_EACH_ACTIVATION_OP(REGISTER_ACTIVATION_OP);
FOR_EACH_ACTIVATION_OP(REGISTER_ACTIVATION_CPU_KERNEL);

REGISTER_OPERATOR(
    relu, ops::ActivationOp, ops::ReluOpMaker, ops::ActivationOpInferVarType,
    ops::ActivationGradOpDescMaker<ops::ReluGradFunctor<float>::FwdDeps()>,
    paddle::framework::SingleOpInplaceInToOut);
REGISTER_OPERATOR(relu_grad, ops::ActivationOpGrad,
                  paddle::framework::SingleOpInplaceInToOut,
                  ops::ReluDoubleGradMaker);
REGISTER_OPERATOR(relu_grad_grad, ops::ActivationOpDoubleGrad);

REGISTER_ACTIVATION_CPU_KERNEL(relu, Relu, ReluFunctor, ReluGradFunctor);

REGISTER_OP_CPU_KERNEL(
    relu_grad_grad,
    ops::ActivationDoubleGradKernel<plat::CPUDeviceContext,
                                    ops::ReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<plat::CPUDeviceContext,
                                    ops::ReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<plat::CPUDeviceContext,
                                    ops::ReluGradGradFunctor<plat::float16>>);
