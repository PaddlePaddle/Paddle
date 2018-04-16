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
#include "paddle/fluid/operators/mkldnn_activation_op.h"

namespace paddle {
namespace operators {

#define REGISTER_ACTIVATION_OP_MAKER(OP_NAME, OP_COMMENT)               \
  class OP_NAME##OpMaker : public framework::OpProtoAndCheckerMaker {   \
   public:                                                              \
    OP_NAME##OpMaker(OpProto *proto, OpAttrChecker *op_checker)         \
        : framework::OpProtoAndCheckerMaker(proto, op_checker) {        \
      AddInput("X", "Input of " #OP_NAME "operator");                   \
      AddOutput("Out", "Output of" #OP_NAME "operator");                \
      AddAttr<bool>("use_mkldnn",                                       \
                    "(bool, default false) Only used in mkldnn kernel") \
          .SetDefault(false);                                           \
      AddComment(#OP_COMMENT);                                          \
    }                                                                   \
  }

#define REGISTER_ACTIVATION_OP_GRAD_MAKER(OP_NAME)                     \
  class OP_NAME##GradMaker : public framework::SingleGradOpDescMaker { \
   public:                                                             \
   protected:                                                          \
    std::unique_ptr<framework::OpDesc> Apply() const override {        \
      auto *op = new framework::OpDesc();                              \
      op->SetType(#OP_NAME "_grad");                                   \
      op->SetInput("Out", Input("Out"));                               \
      op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));  \
                                                                       \
      op->SetAttrMap(Attrs());                                         \
                                                                       \
      op->SetOutput(framework::GradVarName("X"), InputGrad("X"));      \
      return std::unique_ptr<framework::OpDesc>(op);                   \
    }                                                                  \
  }

class ActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Out"));
  }
};

constexpr char SigmoidDoc[] = R"DOC(
Sigmoid Activation Operator

$$out = \frac{1}{1 + e^{-x}}$$

)DOC";

constexpr char LogSigmoidDoc[] = R"DOC(
Logsigmoid Activation Operator

$$out = \log \frac{1}{1 + e^{-x}}$$

)DOC";

constexpr char ExpDoc[] = R"DOC(
Exp Activation Operator.

$out = e^x$

)DOC";

constexpr char ReluDoc[] = R"DOC(
Relu Activation Operator.

$out = \max(x, 0)$

)DOC";

constexpr char TanhDoc[] = R"DOC(
Tanh Activation Operator.

$$out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

constexpr char TanhShrinkDoc[] = R"DOC(
TanhShrink Activation Operator.

$$out = x - \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

constexpr char SqrtDoc[] = R"DOC(
Sqrt Activation Operator.

$out = \sqrt{x}$

)DOC";

constexpr char AbsDoc[] = R"DOC(
Abs Activation Operator.

$out = |x|$

)DOC";

constexpr char CeilDoc[] = R"DOC(
Ceil Activation Operator.

$out = ceil(x)$

)DOC";

constexpr char FloorDoc[] = R"DOC(
Floor Activation Operator.

$out = floor(x)$

)DOC";

constexpr char CosDoc[] = R"DOC(
Cosine Activation Operator.

$out = cos(x)$

)DOC";

constexpr char SinDoc[] = R"DOC(
Sine Activation Operator.

$out = sin(x)$

)DOC";

constexpr char RoundDoc[] = R"DOC(
Round Activation Operator.

$out = [x]$

)DOC";

constexpr char ReciprocalDoc[] = R"DOC(
Reciprocal Activation Operator.

$$out = \frac{1}{x}$$

)DOC";

constexpr char LogDoc[] = R"DOC(
Log Activation Operator.

$out = \ln(x)$

Natural logarithm of x.

)DOC";

constexpr char SquareDoc[] = R"DOC(
Square Activation Operator.

$out = x^2$

)DOC";

constexpr char SoftplusDoc[] = R"DOC(
Softplus Activation Operator.

$out = \ln(1 + e^{x})$

)DOC";

constexpr char SoftsignDoc[] = R"DOC(
Softsign Activation Operator.

$$out = \frac{x}{1 + |x|}$$

)DOC";

class LeakyReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LeakyReluOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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
  SoftShrinkOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Softshrink operator");
    AddOutput("Out", "Output of Softshrink operator");
    AddAttr<float>("lambda", "non-negative offset").SetDefault(0.5f);
    AddComment(R"DOC(
Softshrink Activation Operator.

$$
out = \begin{cases} 
    x - \lambda, \text{if } x > \lambda \\
    x + \lambda, \text{if } x < -\lambda \\
    0,  \text{otherwise}
    \end{cases}
$$

)DOC");
  }
};

class HardShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HardShrinkOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of HardShrink operator");
    AddOutput("Out", "Output of HardShrink operator");
    AddAttr<float>("threshold", "The value of threshold for HardShrink")
        .SetDefault(0.5f);
    AddComment(R"DOC(
HardShrink Activation Operator.

$$
out = \begin{cases} 
    x, \text{if } x > \lambda \\
    x, \text{if } x < -\lambda \\
    0,  \text{otherwise}
    \end{cases}
$$

)DOC");
  }
};

class BReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BReluOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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
  SoftReluOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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
  ELUOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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
  Relu6OpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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
  PowOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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
  STanhOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of STanh operator");
    AddOutput("Out", "Output of STanh operator");
    AddAttr<float>("scale_a", "The scale parameter of a for the input")
        .SetDefault(2.0f / 3.0f);
    AddAttr<float>("scale_b", "The scale parameter of b for the input")
        .SetDefault(1.7159f);
    AddComment(R"DOC(
STanh Activation Operator.

$$out = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}$$

)DOC");
  }
};

class ThresholdedReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ThresholdedReluOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of ThresholdedRelu operator");
    AddOutput("Out", "Output of ThresholdedRelu operator");
    AddAttr<float>("threshold", "The threshold location of activation")
        .SetDefault(1.0f);
    AddComment(R"DOC(
ThresholdedRelu Activation Operator.

$$
out = \begin{cases} 
    x, \text{if } x > threshold \\
    0,  \text{otherwise}
    \end{cases}
$$

)DOC");
  }
};

class HardSigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HardSigmoidOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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
  SwishOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Swish operator");
    AddOutput("Out", "Output of Swish operator");
    AddAttr<float>("beta", "Constant beta of swish operator").SetDefault(1.0f);
    AddComment(R"DOC(
Swish Activation Operator.

$$out = \frac{x}{1 + e^{- \beta x}}$$

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

// NOTE(*) only gradient can be inplaced need to register its gradient maker,
// To tell the executor which input variable is used. By default, every Input
// variable
// is used in gradient operator.
// The operator name written in lowercase intentionally.
REGISTER_ACTIVATION_OP_GRAD_MAKER(sigmoid);
REGISTER_ACTIVATION_OP_GRAD_MAKER(exp);
REGISTER_ACTIVATION_OP_GRAD_MAKER(relu);
REGISTER_ACTIVATION_OP_GRAD_MAKER(tanh);
REGISTER_ACTIVATION_OP_GRAD_MAKER(sqrt);
REGISTER_ACTIVATION_OP_GRAD_MAKER(ceil);
REGISTER_ACTIVATION_OP_GRAD_MAKER(floor);
REGISTER_ACTIVATION_OP_GRAD_MAKER(reciprocal);
REGISTER_ACTIVATION_OP_GRAD_MAKER(relu6);
REGISTER_ACTIVATION_OP_GRAD_MAKER(soft_relu);
REGISTER_ACTIVATION_OP_GRAD_MAKER(hard_sigmoid);
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_ACTIVATION_OP(act_type, op_name)                 \
  REGISTER_OP(act_type, ops::ActivationOp, ops::op_name##OpMaker, \
              act_type##_grad, ops::ActivationOpGrad);

#define FOR_EACH_OP_FUNCTOR(__macro)  \
  __macro(sigmoid, Sigmoid);          \
  __macro(logsigmoid, LogSigmoid);    \
  __macro(exp, Exp);                  \
  __macro(tanh, Tanh);                \
  __macro(softshrink, SoftShrink);    \
  __macro(sqrt, Sqrt);                \
  __macro(abs, Abs);                  \
  __macro(ceil, Ceil);                \
  __macro(floor, Floor);              \
  __macro(cos, Cos);                  \
  __macro(sin, Sin);                  \
  __macro(round, Round);              \
  __macro(reciprocal, Reciprocal);    \
  __macro(log, Log);                  \
  __macro(square, Square);            \
  __macro(brelu, BRelu);              \
  __macro(soft_relu, SoftRelu);       \
  __macro(pow, Pow);                  \
  __macro(stanh, STanh);              \
  __macro(softplus, Softplus);        \
  __macro(softsign, Softsign);        \
  __macro(relu6, Relu6);              \
  __macro(leaky_relu, LeakyRelu);     \
  __macro(tanh_shrink, TanhShrink);   \
  __macro(elu, ELU);                  \
  __macro(hard_shrink, HardShrink);   \
  __macro(hard_sigmoid, HardSigmoid); \
  __macro(swish, Swish);              \
  __macro(thresholded_relu, ThresholdedRelu);

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
FOR_EACH_KERNEL_FUNCTOR(REGISTER_ACTIVATION_CPU_KERNEL);
