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

class SigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SigmoidOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Sigmoid operator");
    AddOutput("Out", "Output of Sigmoid operator");
    AddComment(R"DOC(
Sigmoid Activation Operator

$$out = \frac{1}{1 + e^{-x}}$$

)DOC");
  }
};

class LogSigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LogSigmoidOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of LogSigmoid operator");
    AddOutput("Out", "Output of LogSigmoid operator");
    AddComment(R"DOC(
Logsigmoid Activation Operator

$$out = \log \frac{1}{1 + e^{-x}}$$

)DOC");
  }
};

class ExpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ExpOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Exp operator");
    AddOutput("Out", "Output of Exp operator");
    AddComment(R"DOC(
Exp Activation Operator.

$out = e^x$

)DOC");
  }
};

class ReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReluOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Relu operator");
    AddOutput("Out", "Output of Relu operator");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
Relu Activation Operator.

$out = \max(x, 0)$

)DOC");
  }
};

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

class TanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TanhOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Tanh operator");
    AddOutput("Out", "Output of Tanh operator");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
Tanh Activation Operator.

$$out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC");
  }
};

class TanhShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TanhShrinkOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of TanhShrink operator");
    AddOutput("Out", "Output of TanhShrink operator");
    AddComment(R"DOC(
TanhShrink Activation Operator.

$$out = x - \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

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

class SqrtOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SqrtOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Sqrt operator");
    AddOutput("Out", "Output of Sqrt operator");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
Sqrt Activation Operator.

$out = \sqrt{x}$

)DOC");
  }
};

class AbsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AbsOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Abs operator");
    AddOutput("Out", "Output of Abs operator");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
Abs Activation Operator.

$out = |x|$

)DOC");
  }
};

class CeilOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CeilOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Ceil operator");
    AddOutput("Out", "Output of Ceil operator");
    AddComment(R"DOC(
Ceil Activation Operator.

$out = ceil(x)$

)DOC");
  }
};

class FloorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FloorOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Floor operator");
    AddOutput("Out", "Output of Floor operator");
    AddComment(R"DOC(
Floor Activation Operator.

$out = floor(x)$

)DOC");
  }
};

class CosOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CosOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Cosine operator");
    AddOutput("Out", "Output of Cosine operator");
    AddComment(R"DOC(
Cosine Activation Operator.

$out = cos(x)$

)DOC");
  }
};

class SinOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SinOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Sine operator");
    AddOutput("Out", "Output of Sine operator");
    AddComment(R"DOC(
Sine Activation Operator.

$out = sin(x)$

)DOC");
  }
};

class RoundOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RoundOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Round operator");
    AddOutput("Out", "Output of Round operator");
    AddComment(R"DOC(
Round Activation Operator.

$out = [x]$

)DOC");
  }
};

class ReciprocalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReciprocalOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Reciprocal operator");
    AddOutput("Out", "Output of Reciprocal operator");
    AddComment(R"DOC(
Reciprocal Activation Operator.

$$out = \frac{1}{x}$$

)DOC");
  }
};

class LogOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LogOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Log operator");
    AddOutput("Out", "Output of Log operator");
    AddComment(R"DOC(
Log Activation Operator.

$out = \ln(x)$

Natural logarithm of x.

)DOC");
  }
};

class SquareOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SquareOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Square operator");
    AddOutput("Out", "Output of Square operator");
    AddComment(R"DOC(
Square Activation Operator.

$out = x^2$

)DOC");
  }
};

class SoftplusOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftplusOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Softplus operator");
    AddOutput("Out", "Output of Softplus operator");
    AddComment(R"DOC(
Softplus Activation Operator.

$out = \ln(1 + e^{x})$

)DOC");
  }
};

class SoftsignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftsignOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Softsign operator");
    AddOutput("Out", "Output of Softsign operator");
    AddComment(R"DOC(
Softsign Activation Operator.

$$out = \frac{x}{1 + |x|}$$

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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(sigmoid, ops::ActivationOp, ops::SigmoidOpMaker, sigmoid_grad,
            ops::ActivationOpGrad);

REGISTER_OP(logsigmoid, ops::ActivationOp, ops::LogSigmoidOpMaker,
            logsigmoid_grad, ops::ActivationOpGrad);

REGISTER_OP(exp, ops::ActivationOp, ops::ExpOpMaker, exp_grad,
            ops::ActivationOpGrad);

REGISTER_OP(relu, ops::ActivationWithMKLDNNOp, ops::ReluOpMaker, relu_grad,
            ops::ActivationWithMKLDNNOpGrad);

REGISTER_OP(tanh, ops::ActivationWithMKLDNNOp, ops::TanhOpMaker, tanh_grad,
            ops::ActivationWithMKLDNNOpGrad);

REGISTER_OP(tanh_shrink, ops::ActivationOp, ops::TanhShrinkOpMaker,
            tanh_shrink_grad, ops::ActivationOpGrad);

REGISTER_OP(softshrink, ops::ActivationOp, ops::SoftShrinkOpMaker,
            softshrink_grad, ops::ActivationOpGrad);

REGISTER_OP(sqrt, ops::ActivationWithMKLDNNOp, ops::SqrtOpMaker, sqrt_grad,
            ops::ActivationWithMKLDNNOpGrad);

REGISTER_OP(abs, ops::ActivationWithMKLDNNOp, ops::AbsOpMaker, abs_grad,
            ops::ActivationWithMKLDNNOpGrad);

REGISTER_OP(ceil, ops::ActivationOp, ops::CeilOpMaker, ceil_grad,
            ops::ActivationOpGrad);

REGISTER_OP(floor, ops::ActivationOp, ops::FloorOpMaker, floor_grad,
            ops::ActivationOpGrad);

REGISTER_OP(cos, ops::ActivationOp, ops::CosOpMaker, cos_grad,
            ops::ActivationOpGrad);

REGISTER_OP(sin, ops::ActivationOp, ops::SinOpMaker, sin_grad,
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

FOR_EACH_KERNEL_FUNCTOR(REGISTER_ACTIVATION_CPU_KERNEL);

REGISTER_OP_CPU_KERNEL(relu,
                       ops::ActivationKernel<paddle::platform::CPUDeviceContext,
                                             ops::ReluFunctor<float>>,
                       ops::ActivationKernel<paddle::platform::CPUDeviceContext,
                                             ops::ReluFunctor<double>>);
REGISTER_OP_CPU_KERNEL(
    relu_grad, ops::ActivationGradKernel<paddle::platform::CPUDeviceContext,
                                         ops::ReluGradFunctor<float>>,
    ops::ActivationGradKernel<paddle::platform::CPUDeviceContext,
                              ops::ReluGradFunctor<double>>);
