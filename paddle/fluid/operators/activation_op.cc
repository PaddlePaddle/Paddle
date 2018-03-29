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

template <bool use_mkldnn = false>
class BaseActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    if (use_mkldnn) {
      framework::LibraryType library{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_MKLDNN
      if (library == framework::LibraryType::kPlain &&
          platform::CanMKLDNNBeUsed(ctx)) {
        library = framework::LibraryType::kMKLDNN;
      }
#endif
      framework::DataLayout layout = framework::DataLayout::kAnyLayout;
      return framework::OpKernelType(
          framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
          ctx.GetPlace(), layout, library);
    } else {
      return framework::OperatorWithKernel::GetExpectedKernelType(ctx);
    }
  }
};

template <bool use_mkldnn>
class ActivationOp : public BaseActivationOp<use_mkldnn> {
 public:
  using BaseActivationOp<use_mkldnn>::BaseActivationOp;
  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <bool use_mkldnn = false>
class ActivationOpGrad : public BaseActivationOp<use_mkldnn> {
 public:
  using BaseActivationOp<use_mkldnn>::BaseActivationOp;
  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Out"));
  }
};

template <typename T>
class ActivationOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ActivationOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", string::Sprintf("Input of %s operator", T::Type));
    AddOutput("Out", string::Sprintf("Output of %s operator", T::Type));
    AddComment(string::Sprintf(R"DOC(
%s Activation Operator

$$%s$$

)DOC",
                               T::Type, T::Equation));
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_ACTIVATION_EX(op_type, maker, use_mkldnn)                   \
  REGISTER_OP(op_type, ops::ActivationOp<use_mkldnn>, maker, op_type##_grad, \
              ops::ActivationOpGrad<use_mkldnn>)

#define REGISTER_ACTIVATION_WITH_EQUATION(op_type, equation, use_mkldnn) \
  struct op_type##OpMakerTrait {                                         \
    static constexpr char Type[] = #op_type;                             \
    static constexpr char Equation[] = equation;                         \
  };                                                                     \
  using op_type##Maker = ops::ActivationOpMaker<op_type##OpMakerTrait>;  \
  REGISTER_ACTIVATION_EX(op_type, op_type##Maker, use_mkldnn)

#define REGISTER_ACTIVATION(op_type, equation) \
  REGISTER_ACTIVATION_WITH_EQUATION(op_type, equation, false)

#define REGISTER_MKLDNN_ACTIVATION(op_type, equation) \
  REGISTER_ACTIVATION_WITH_EQUATION(op_type, equation, true)

REGISTER_ACTIVATION(sigmoid, R"DOC(out = \\frac{1}{1 + e^{-x}})DOC");
REGISTER_ACTIVATION(logsigmoid, R"DOC(out = \\log \\frac{1}{1 + e^{-x}})DOC");
REGISTER_ACTIVATION(exp, R"DOC(out = e^x)DOC");
REGISTER_MKLDNN_ACTIVATION(relu, R"DOC(out = \max(x, 0))DOC");
REGISTER_MKLDNN_ACTIVATION(
    tanh, R"DOC(out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}})DOC");
REGISTER_ACTIVATION(tanh_shrink,
                    R"DOC(out = x - \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}})DOC");
REGISTER_ACTIVATION(softshrink, R"DOC(out = \begin{cases}
    x - \lambda, \text{if } x > \lambda \\
    x + \lambda, \text{if } x < -\lambda \\
    0,  \text{otherwise}
    \end{cases})DOC");

REGISTER_MKLDNN_ACTIVATION(sqrt, R"DOC(out = \sqrt{x})DOC");
REGISTER_MKLDNN_ACTIVATION(abs, R"DOC(out = |x|)DOC");
REGISTER_ACTIVATION(ceil, R"DOC(out = ceil(x))DOC");
REGISTER_ACTIVATION(floor, R"DOC(out = floor(x))DOC");
REGISTER_ACTIVATION(cos, R"DOC(out = cos(x))DOC");
REGISTER_ACTIVATION(sin, R"DOC(out = sin(x))DOC");
REGISTER_ACTIVATION(round, R"DOC(out = [x])DOC");
REGISTER_ACTIVATION(reciprocal, R"DOC(out = \frac{1}{x})DOC");
REGISTER_ACTIVATION(log, R"DOC(out = log(x))DOC");
REGISTER_ACTIVATION(square, R"DOC(out = x^2)DOC");
REGISTER_ACTIVATION(softplus, R"DOC(out = \ln(1 + e^{x}))DOC");
REGISTER_ACTIVATION(softsign, R"DOC(out = \frac{x}{1 + |x|})DOC");
REGISTER_ACTIVATION_EX(brelu, ops::BReluOpMaker, false);
REGISTER_ACTIVATION_EX(leaky_relu, ops::LeakyReluOpMaker, false);
REGISTER_ACTIVATION_EX(soft_relu, ops::SoftReluOpMaker, false);
REGISTER_ACTIVATION_EX(elu, ops::ELUOpMaker, false);
REGISTER_ACTIVATION_EX(relu6, ops::Relu6OpMaker, false);
REGISTER_ACTIVATION_EX(pow, ops::PowOpMaker, false);
REGISTER_ACTIVATION_EX(stanh, ops::STanhOpMaker, false);
REGISTER_ACTIVATION_EX(hard_shrink, ops::HardShrinkOpMaker, false);
REGISTER_ACTIVATION_EX(thresholded_relu, ops::ThresholdedReluOpMaker, false);
REGISTER_ACTIVATION_EX(hard_sigmoid, ops::HardSigmoidOpMaker, false);
REGISTER_ACTIVATION_EX(swish, ops::SwishOpMaker, false);

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
