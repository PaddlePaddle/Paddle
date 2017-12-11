/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/activation_op.h"

namespace paddle {
namespace operators {

class ActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Y");
  }
};

class ActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Y"));
  }
};

class SigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SigmoidOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Sigmoid operator");
    AddOutput("Y", "Output of Sigmoid operator");
    AddComment(R"DOC(
Sigmoid Activation Operator

$$y = \frac{1}{1 + e^{-x}}$$

)DOC");
  }
};

class LogSigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LogSigmoidOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of LogSigmoid operator");
    AddOutput("Y", "Output of LogSigmoid operator");
    AddComment(R"DOC(
Logsigmoid Activation Operator

$$y = \log \frac{1}{1 + e^{-x}}$$

)DOC");
  }
};

class ExpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ExpOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Exp operator");
    AddOutput("Y", "Output of Exp operator");
    AddComment(R"DOC(
Exp Activation Operator.

$y = e^x$

)DOC");
  }
};

class ReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReluOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Relu operator");
    AddOutput("Y", "Output of Relu operator");
    AddComment(R"DOC(
Relu Activation Operator.

$y = \max(x, 0)$

)DOC");
  }
};

class LeakyReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LeakyReluOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of LeakyRelu operator");
    AddOutput("Y", "Output of LeakyRelu operator");
    AddAttr<float>("alpha", "The small negative slope").SetDefault(0.02f);
    AddComment(R"DOC(
LeakyRelu Activation Operator.

$y = \max(x, \alpha * x)$

)DOC");
  }
};

class SoftShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftShrinkOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Softshrink operator");
    AddOutput("Y", "Output of Softshrink operator");
    AddAttr<float>("lambda", "non-negative offset").SetDefault(0.5f);
    AddComment(R"DOC(
Softshrink Activation Operator.

$$
y = \begin{cases} 
    x - \lambda, \text{if } x > \lambda \\
    x + \lambda, \text{if } x < -\lambda \\
    0,  \text{otherwise}
    \end{cases}
$$

)DOC");
  }
};

class TanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TanhOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Tanh operator");
    AddOutput("Y", "Output of Tanh operator");
    AddComment(R"DOC(
Tanh Activation Operator.

$$y = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC");
  }
};

class TanhShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TanhShrinkOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of TanhShrink operator");
    AddOutput("Y", "Output of TanhShrink operator");
    AddComment(R"DOC(
TanhShrink Activation Operator.

$$y = x - \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC");
  }
};

class HardShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HardShrinkOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of HardShrink operator");
    AddOutput("Y", "Output of HardShrink operator");
    AddAttr<float>("threshold", "The value of threshold for HardShrink")
        .SetDefault(0.5f);
    AddComment(R"DOC(
HardShrink Activation Operator.

$$
y = \begin{cases} 
    x, \text{if } x > \lambda \\
    x, \text{if } x < -\lambda \\
    0,  \text{otherwise}
    \end{cases}
$$

)DOC");
  }
};

class SqrtOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SqrtOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Sqrt operator");
    AddOutput("Y", "Output of Sqrt operator");
    AddComment(R"DOC(
Sqrt Activation Operator.

$y = \sqrt{x}$

)DOC");
  }
};

class AbsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AbsOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Abs operator");
    AddOutput("Y", "Output of Abs operator");
    AddComment(R"DOC(
Abs Activation Operator.

$y = |x|$

)DOC");
  }
};

class CeilOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CeilOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Ceil operator");
    AddOutput("Y", "Output of Ceil operator");
    AddComment(R"DOC(
Ceil Activation Operator.

$y = ceil(x)$

)DOC");
  }
};

class FloorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FloorOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Floor operator");
    AddOutput("Y", "Output of Floor operator");
    AddComment(R"DOC(
Floor Activation Operator.

$y = floor(x)$

)DOC");
  }
};

class RoundOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RoundOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Round operator");
    AddOutput("Y", "Output of Round operator");
    AddComment(R"DOC(
Round Activation Operator.

$y = [x]$

)DOC");
  }
};

class ReciprocalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReciprocalOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Reciprocal operator");
    AddOutput("Y", "Output of Reciprocal operator");
    AddComment(R"DOC(
Reciprocal Activation Operator.

$$y = \frac{1}{x}$$

)DOC");
  }
};

class LogOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LogOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Log operator");
    AddOutput("Y", "Output of Log operator");
    AddComment(R"DOC(
Log Activation Operator.

$y = \ln(x)$

Natural logarithm of x.

)DOC");
  }
};

class SquareOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SquareOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Square operator");
    AddOutput("Y", "Output of Square operator");
    AddComment(R"DOC(
Square Activation Operator.

$y = x^2$

)DOC");
  }
};

class SoftplusOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftplusOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Softplus operator");
    AddOutput("Y", "Output of Softplus operator");
    AddComment(R"DOC(
Softplus Activation Operator.

$y = \ln(1 + e^{x})$

)DOC");
  }
};

class SoftsignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftsignOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Softsign operator");
    AddOutput("Y", "Output of Softsign operator");
    AddComment(R"DOC(
Softsign Activation Operator.

$$y = \frac{x}{1 + |x|}$$

)DOC");
  }
};

class BReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BReluOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of BRelu operator");
    AddOutput("Y", "Output of BRelu operator");
    AddAttr<float>("t_min", "The min marginal value of BRelu")
        .SetDefault(static_cast<float>(0));
    AddAttr<float>("t_max", "The max marginal value of BRelu")
        .SetDefault(static_cast<float>(24));
    AddComment(R"DOC(
BRelu Activation Operator.

$y = \max(\min(x, t_{min}), t_{max})$

)DOC");
  }
};

class SoftReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftReluOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of SoftRelu operator");
    AddOutput("Y", "Output of SoftRelu operator");
    AddAttr<float>("threshold", "The threshold value of SoftRelu")
        .SetDefault(40.0f);
    AddComment(R"DOC(
SoftRelu Activation Operator.

$y = \ln(1 + \exp(\max(\min(x, threshold), threshold))$

)DOC");
  }
};

class ELUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ELUOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of ELU operator");
    AddOutput("Y", "Output of ELU operator");
    AddAttr<float>("alpha", "The alpha value of ELU").SetDefault(1.0f);
    AddComment(R"DOC(
ELU Activation Operator.

Applies the following element-wise computation on the input according to
https://arxiv.org/abs/1511.07289.

$y = \max(0, x) + \min(0, \alpha * (e^x - 1))$

)DOC");
  }
};

class Relu6OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Relu6OpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Relu6 operator");
    AddOutput("Y", "Output of Relu6 operator");
    AddAttr<float>("threshold", "The threshold value of Relu6")
        .SetDefault(6.0f);
    AddComment(R"DOC(
Relu6 Activation Operator.

$y = \min(\max(0, x), 6)$

)DOC");
  }
};

class PowOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PowOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Pow operator");
    AddOutput("Y", "Output of Pow operator");
    AddAttr<float>("factor", "The exponential factor of Pow").SetDefault(1.0f);
    AddComment(R"DOC(
Pow Activation Operator.

$y = x^{factor}$

)DOC");
  }
};

class STanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  STanhOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of STanh operator");
    AddOutput("Y", "Output of STanh operator");
    AddAttr<float>("scale_a", "The scale parameter of a for the input")
        .SetDefault(2.0f / 3.0f);
    AddAttr<float>("scale_b", "The scale parameter of b for the input")
        .SetDefault(1.7159f);
    AddComment(R"DOC(
STanh Activation Operator.

$$y = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}$$

)DOC");
  }
};

class ThresholdedReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ThresholdedReluOpMaker(framework::OpProto *proto,
                         framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of ThresholdedRelu operator");
    AddOutput("Y", "Output of ThresholdedRelu operator");
    AddAttr<float>("threshold", "The threshold location of activation")
        .SetDefault(1.0f);
    AddComment(R"DOC(
ThresholdedRelu Activation Operator.

$$
y = \begin{cases} 
    x, \text{if } x > threshold \\
    0,  \text{otherwise}
    \end{cases}
$$

)DOC");
  }
};

class HardSigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HardSigmoidOpMaker(framework::OpProto *proto,
                     framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of HardSigmoid operator");
    AddOutput("Y", "Output of HardSigmoid operator");
    AddAttr<float>("slope", "Slope for linear approximation of sigmoid")
        .SetDefault(0.2f);
    AddAttr<float>("offset", "Offset for linear approximation of sigmoid")
        .SetDefault(0.5f);
    AddComment(R"DOC(
HardSigmoid Activation Operator.

Segment-wise linear approximation of sigmoid(https://arxiv.org/abs/1603.00391), 
which is much faster than sigmoid.

$y = \max(0, \min(1, slope * x + shift))$

The slope should be positive. The offset can be either positive or negative.
The default slope and shift are set according to the above reference.
It is recommended to use the defaults for this activation.

)DOC");
  }
};

class SwishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SwishOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Swish operator");
    AddOutput("Y", "Output of Swish operator");
    AddAttr<float>("beta", "Constant beta of swish operator").SetDefault(1.0f);
    AddComment(R"DOC(
Swish Activation Operator.

$$y = \frac{x}{1 + e^{- \beta x}}$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(sigmoid, ops::ActivationOp, ops::SigmoidOpMaker, sigmoid_grad,
            ops::ActivationOpGrad);

REGISTER_OP(logsigmoid, ops::ActivationOp, ops::LogSigmoidOpMaker,
            logsigmoid_grad, ops::ActivationOpGrad);

REGISTER_OP(exp, ops::ActivationOp, ops::ExpOpMaker, exp_grad,
            ops::ActivationOpGrad);

REGISTER_OP(relu, ops::ActivationOp, ops::ReluOpMaker, relu_grad,
            ops::ActivationOpGrad);

REGISTER_OP(tanh, ops::ActivationOp, ops::TanhOpMaker, tanh_grad,
            ops::ActivationOpGrad);

REGISTER_OP(tanh_shrink, ops::ActivationOp, ops::TanhShrinkOpMaker,
            tanh_shrink_grad, ops::ActivationOpGrad);

REGISTER_OP(softshrink, ops::ActivationOp, ops::SoftShrinkOpMaker,
            softshrink_grad, ops::ActivationOpGrad);

REGISTER_OP(sqrt, ops::ActivationOp, ops::SqrtOpMaker, sqrt_grad,
            ops::ActivationOpGrad);

REGISTER_OP(abs, ops::ActivationOp, ops::AbsOpMaker, abs_grad,
            ops::ActivationOpGrad);

REGISTER_OP(ceil, ops::ActivationOp, ops::CeilOpMaker, ceil_grad,
            ops::ActivationOpGrad);

REGISTER_OP(floor, ops::ActivationOp, ops::FloorOpMaker, floor_grad,
            ops::ActivationOpGrad);

REGISTER_OP(round, ops::ActivationOp, ops::RoundOpMaker, round_grad,
            ops::ActivationOpGrad);

REGISTER_OP(reciprocal, ops::ActivationOp, ops::ReciprocalOpMaker,
            reciprocal_grad, ops::ActivationOpGrad);

REGISTER_OP(log, ops::ActivationOp, ops::LogOpMaker, log_grad,
            ops::ActivationOpGrad);

REGISTER_OP(square, ops::ActivationOp, ops::SquareOpMaker, square_grad,
            ops::ActivationOpGrad);

REGISTER_OP(softplus, ops::ActivationOp, ops::SoftplusOpMaker, softplus_grad,
            ops::ActivationOpGrad);

REGISTER_OP(softsign, ops::ActivationOp, ops::SoftsignOpMaker, softsign_grad,
            ops::ActivationOpGrad);

REGISTER_OP(brelu, ops::ActivationOp, ops::BReluOpMaker, brelu_grad,
            ops::ActivationOpGrad);

REGISTER_OP(leaky_relu, ops::ActivationOp, ops::LeakyReluOpMaker,
            leaky_relu_grad, ops::ActivationOpGrad);

REGISTER_OP(soft_relu, ops::ActivationOp, ops::SoftReluOpMaker, soft_relu_grad,
            ops::ActivationOpGrad);

REGISTER_OP(elu, ops::ActivationOp, ops::ELUOpMaker, elu_grad,
            ops::ActivationOpGrad);

REGISTER_OP(relu6, ops::ActivationOp, ops::Relu6OpMaker, relu6_grad,
            ops::ActivationOpGrad);

REGISTER_OP(pow, ops::ActivationOp, ops::PowOpMaker, pow_grad,
            ops::ActivationOpGrad);

REGISTER_OP(stanh, ops::ActivationOp, ops::STanhOpMaker, stanh_grad,
            ops::ActivationOpGrad);

REGISTER_OP(hard_shrink, ops::ActivationOp, ops::HardShrinkOpMaker,
            hard_shrink_grad, ops::ActivationOpGrad);

REGISTER_OP(thresholded_relu, ops::ActivationOp, ops::ThresholdedReluOpMaker,
            thresholded_relu_grad, ops::ActivationOpGrad);

REGISTER_OP(hard_sigmoid, ops::ActivationOp, ops::HardSigmoidOpMaker,
            hard_sigmoid_grad, ops::ActivationOpGrad);

REGISTER_OP(swish, ops::ActivationOp, ops::SwishOpMaker, swish_grad,
            ops::ActivationOpGrad);

#define REGISTER_ACTIVATION_CPU_KERNEL(act_type, functor, grad_functor)       \
  REGISTER_OP_CPU_KERNEL(                                                     \
      act_type,                                                               \
      ops::ActivationKernel<paddle::platform::CPUPlace, ops::functor<float>>, \
      ops::ActivationKernel<paddle::platform::CPUPlace,                       \
                            ops::functor<double>>);                           \
  REGISTER_OP_CPU_KERNEL(                                                     \
      act_type##_grad, ops::ActivationGradKernel<paddle::platform::CPUPlace,  \
                                                 ops::grad_functor<float>>,   \
      ops::ActivationGradKernel<paddle::platform::CPUPlace,                   \
                                ops::grad_functor<double>>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_ACTIVATION_CPU_KERNEL);
