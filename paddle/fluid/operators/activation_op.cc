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
#include <string>
#include "paddle/fluid/operators/mkldnn_activation_op.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

#define REGISTER_ACTIVATION_OP_MAKER(OP_NAME, OP_COMMENT)               \
  class OP_NAME##OpMaker                                                \
      : public ::paddle::framework::OpProtoAndCheckerMaker {            \
   public:                                                              \
    void Make() override {                                              \
      AddInput("X", "Input of " #OP_NAME " operator");                  \
      AddOutput("Out", "Output of " #OP_NAME " operator").Reuse("X");   \
      AddAttr<bool>("use_mkldnn",                                       \
                    "(bool, default false) Only used in mkldnn kernel") \
          .SetDefault(false);                                           \
      AddComment(#OP_COMMENT);                                          \
    }                                                                   \
  }

#define REGISTER_ACTIVATION_OP_GRAD_MAKER(OP_NAME, KERNEL_TYPE)              \
  class OP_NAME##GradMaker                                                   \
      : public ::paddle::framework::SingleGradOpDescMaker {                  \
   public:                                                                   \
    using ::paddle::framework::SingleGradOpDescMaker::SingleGradOpDescMaker; \
                                                                             \
   protected:                                                                \
    std::unique_ptr<::paddle::framework::OpDesc> Apply() const override {    \
      auto* op = new ::paddle::framework::OpDesc();                          \
      op->SetType(#KERNEL_TYPE "_grad");                                     \
      op->SetInput("Out", Output("Out"));                                    \
      op->SetInput(::paddle::framework::GradVarName("Out"),                  \
                   OutputGrad("Out"));                                       \
                                                                             \
      op->SetAttrMap(Attrs());                                               \
                                                                             \
      op->SetOutput(::paddle::framework::GradVarName("X"), InputGrad("X"));  \
      return std::unique_ptr<::paddle::framework::OpDesc>(op);               \
    }                                                                        \
  }

framework::OpKernelType GetKernelType(const framework::ExecutionContext& ctx,
                                      const framework::OperatorWithKernel& oper,
                                      const std::string& name) {
  framework::LibraryType library{framework::LibraryType::kPlain};
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
#ifdef PADDLE_WITH_MKLDNN
  auto it = oper.Attrs().find("use_mkldnn");
  if (library == framework::LibraryType::kPlain && it != oper.Attrs().end() &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::Tensor>(name)->type()),
      ctx.GetPlace(), layout, library);
}

class ActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "X");
  }
};

class ActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Out"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "Out");
  }
};

__attribute__((unused)) constexpr char SigmoidDoc[] = R"DOC(
Sigmoid Activation Operator

$$out = \frac{1}{1 + e^{-x}}$$

)DOC";

__attribute__((unused)) constexpr char LogSigmoidDoc[] = R"DOC(
Logsigmoid Activation Operator

$$out = \\log \\frac{1}{1 + e^{-x}}$$

)DOC";

__attribute__((unused)) constexpr char ExpDoc[] = R"DOC(
Exp Activation Operator.

$out = e^x$

)DOC";

__attribute__((unused)) constexpr char ReluDoc[] = R"DOC(
Relu Activation Operator.

$out = \max(x, 0)$

)DOC";

__attribute__((unused)) constexpr char TanhDoc[] = R"DOC(
Tanh Activation Operator.

$$out = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

__attribute__((unused)) constexpr char TanhShrinkDoc[] = R"DOC(
TanhShrink Activation Operator.

$$out = x - \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

__attribute__((unused)) constexpr char SqrtDoc[] = R"DOC(
Sqrt Activation Operator.

$out = \sqrt{x}$

)DOC";

__attribute__((unused)) constexpr char AbsDoc[] = R"DOC(
Abs Activation Operator.

$out = |x|$

)DOC";

__attribute__((unused)) constexpr char CeilDoc[] = R"DOC(
Ceil Activation Operator.

$out = ceil(x)$

)DOC";

__attribute__((unused)) constexpr char FloorDoc[] = R"DOC(
Floor Activation Operator.

$out = floor(x)$

)DOC";

__attribute__((unused)) constexpr char CosDoc[] = R"DOC(
Cosine Activation Operator.

$out = cos(x)$

)DOC";

__attribute__((unused)) constexpr char SinDoc[] = R"DOC(
Sine Activation Operator.

$out = sin(x)$

)DOC";

__attribute__((unused)) constexpr char RoundDoc[] = R"DOC(
Round Activation Operator.

$out = [x]$

)DOC";

__attribute__((unused)) constexpr char ReciprocalDoc[] = R"DOC(
Reciprocal Activation Operator.

$$out = \\frac{1}{x}$$

)DOC";

__attribute__((unused)) constexpr char LogDoc[] = R"DOC(
Log Activation Operator.

$out = \ln(x)$

Natural logarithm of x.

)DOC";

__attribute__((unused)) constexpr char SquareDoc[] = R"DOC(
Square Activation Operator.

$out = x^2$

)DOC";

__attribute__((unused)) constexpr char SoftplusDoc[] = R"DOC(
Softplus Activation Operator.

$out = \ln(1 + e^{x})$

)DOC";

__attribute__((unused)) constexpr char SoftsignDoc[] = R"DOC(
Softsign Activation Operator.

$$out = \frac{x}{1 + |x|}$$

)DOC";

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

REGISTER_ACTIVATION_OP_GRAD_MAKER(Sigmoid, sigmoid);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Relu, relu);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Exp, exp);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Tanh, tanh);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Ceil, ceil);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Floor, floor);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Sqrt, sqrt);
REGISTER_ACTIVATION_OP_GRAD_MAKER(SoftRelu, soft_relu);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Relu6, relu6);
REGISTER_ACTIVATION_OP_GRAD_MAKER(Reciprocal, reciprocal);
REGISTER_ACTIVATION_OP_GRAD_MAKER(HardSigmoid, hard_sigmoid);
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define FOR_EACH_INPLACE_OP_FUNCTOR(__macro) \
  __macro(Sigmoid, sigmoid);                 \
  __macro(Relu, relu);                       \
  __macro(Exp, exp);                         \
  __macro(Tanh, tanh);                       \
  __macro(Ceil, ceil);                       \
  __macro(Floor, floor);                     \
  __macro(Sqrt, sqrt);                       \
  __macro(SoftRelu, soft_relu);              \
  __macro(Relu6, relu6);                     \
  __macro(Reciprocal, reciprocal);           \
  __macro(HardSigmoid, hard_sigmoid);

#define FOR_EACH_OP_FUNCTOR(__macro) \
  __macro(LogSigmoid, logsigmoid);   \
  __macro(SoftShrink, softshrink);   \
  __macro(Abs, abs);                 \
  __macro(Cos, cos);                 \
  __macro(Sin, sin);                 \
  __macro(Round, round);             \
  __macro(Log, log);                 \
  __macro(Square, square);           \
  __macro(BRelu, brelu);             \
  __macro(Pow, pow);                 \
  __macro(STanh, stanh);             \
  __macro(Softplus, softplus);       \
  __macro(Softsign, softsign);       \
  __macro(LeakyRelu, leaky_relu);    \
  __macro(TanhShrink, tanh_shrink);  \
  __macro(ELU, elu);                 \
  __macro(HardShrink, hard_shrink);  \
  __macro(Swish, swish);             \
  __macro(ThresholdedRelu, thresholded_relu);

#define REGISTER_INPLACE_ACTIVATION_OP(OP_NAME, KERNEL_TYPE)        \
  REGISTER_OPERATOR(KERNEL_TYPE, ::paddle::operators::ActivationOp, \
                    ::paddle::operators::OP_NAME##OpMaker,          \
                    ::paddle::operators::OP_NAME##GradMaker);       \
  REGISTER_OPERATOR(KERNEL_TYPE##_grad, ::paddle::operators::ActivationOpGrad)

#define REGISTER_ACTIVATION_OP(OP_NAME, KERNEL_TYPE)                    \
  REGISTER_OPERATOR(KERNEL_TYPE, ::paddle::operators::ActivationOp,     \
                    ::paddle::operators::OP_NAME##OpMaker,              \
                    ::paddle::framework::DefaultGradOpDescMaker<true>); \
  REGISTER_OPERATOR(KERNEL_TYPE##_grad, ::paddle::operators::ActivationOpGrad)

#define REGISTER_ACTIVATION_CPU_KERNEL(act_type, functor, grad_functor)   \
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

FOR_EACH_OP_FUNCTOR(REGISTER_ACTIVATION_OP);
FOR_EACH_INPLACE_OP_FUNCTOR(REGISTER_INPLACE_ACTIVATION_OP);
FOR_EACH_KERNEL_FUNCTOR(REGISTER_ACTIVATION_CPU_KERNEL);
