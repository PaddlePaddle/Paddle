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

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Y");
  }
};

class ActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
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
    AddComment("Sigmoid activation operator, sigmoid = 1 / (1 + exp(-x))");
  }
};

class ExpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ExpOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Exp operator");
    AddOutput("Y", "Output of Exp operator");
    AddComment("Exp activation operator, exp(x) = e^x");
  }
};

class ReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReluOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Relu operator");
    AddOutput("Y", "Output of Relu operator");
    AddComment("Relu activation operator, relu(x) = max(x, 0)");
  }
};

template <typename AttrType>
class LeakyReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LeakyReluOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of LeakyRelu operator");
    AddOutput("Y", "Output of LeakyRelu operator");
    AddComment(
        "LeakyRelu activation operator, "
        "leaky_relu = max(x, alpha * x)");
    AddAttr<AttrType>("alpha", "The small negative slope")
        .SetDefault(static_cast<AttrType>(0.02f));
  }
};

class TanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TanhOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Tanh operator");
    AddOutput("Y", "Output of Tanh operator");
    AddComment(
        "Tanh activation operator, tanh = (exp(x) - exp(-x)) / (exp(x) + "
        "exp(-x))");
  }
};

class TanhShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TanhShrinkOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of TanhShrink operator");
    AddOutput("Y", "Output of TanhShrink operator");
    AddComment("TanhShrink activation operator, tanhshrink(x) = x - tanh(x)");
  }
};

class SqrtOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SqrtOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Sqrt operator");
    AddOutput("Y", "Output of Sqrt operator");
    AddComment("Sqrt activation operator, sqrt(x) = x^(1/2)");
  }
};

class AbsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AbsOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Abs operator");
    AddOutput("Y", "Output of Abs operator");
    AddComment("Abs activation operator, abs(x) = |x|");
  }
};

class ReciprocalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReciprocalOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Reciprocal operator");
    AddOutput("Y", "Output of Reciprocal operator");
    AddComment("Reciprocal activation operator, reciprocal(x) = 1 / x");
  }
};

class LogOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LogOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Log operator");
    AddOutput("Y", "Output of Log operator");
    AddComment("Log activation operator, log(x) = natural logarithm of x");
  }
};

class SquareOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SquareOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Square operator");
    AddOutput("Y", "Output of Square operator");
    AddComment("Square activation operator, square(x) = x^2");
  }
};

class SoftsignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftsignOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Softsign operator");
    AddOutput("Y", "Output of Softsign operator");
    AddComment("Softsign activation operator, softsign(x) = x / (1 + |x|)");
  }
};

template <typename AttrType>
class BReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BReluOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of BRelu operator");
    AddOutput("Y", "Output of BRelu operator");
    AddComment("BRelu activation operator, brelu = max(min(x, t_min), t_max)");
    AddAttr<AttrType>("t_min", "The min marginal value of BRelu")
        .SetDefault(static_cast<AttrType>(0));
    AddAttr<AttrType>("t_max", "The max marginal value of BRelu")
        .SetDefault(static_cast<AttrType>(24));
  }
};

template <typename AttrType>
class SoftReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftReluOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of SoftRelu operator");
    AddOutput("Y", "Output of SoftRelu operator");
    AddComment(
        "SoftRelu activation operator, soft_relu = log(1 + exp(max(min(x, "
        "threshold), threshold)))");
    AddAttr<AttrType>("threshold", "The threshold value of SoftRelu")
        .SetDefault(static_cast<AttrType>(40));
  }
};

template <typename AttrType>
class Relu6OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Relu6OpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Relu6 operator");
    AddOutput("Y", "Output of Relu6 operator");
    AddComment("Relu6 activation operator, relu6 = min(max(0, x), 6)");
    AddAttr<AttrType>("threshold", "The threshold value of Relu6")
        .SetDefault(static_cast<AttrType>(6));
  }
};

template <typename AttrType>
class PowOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PowOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of Pow operator");
    AddOutput("Y", "Output of Pow operator");
    AddComment("Pow activation operator, pow(x, factor) = x^factor");
    AddAttr<AttrType>("factor", "The exponential factor of Pow")
        .SetDefault(static_cast<AttrType>(1));
  }
};

template <typename AttrType>
class STanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  STanhOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of STanh operator");
    AddOutput("Y", "Output of STanh operator");
    AddComment("STanh activation operator, stanh = b * tanh(a * x)");
    AddAttr<AttrType>("scale_a", "The scale parameter of a for the input")
        .SetDefault(static_cast<AttrType>(2 / 3));
    AddAttr<AttrType>("scale_b", "The scale parameter of b for the input")
        .SetDefault(static_cast<AttrType>(1.7159));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(sigmoid, ops::ActivationOp, ops::SigmoidOpMaker, sigmoid_grad,
            ops::ActivationOpGrad);

REGISTER_OP(exp, ops::ActivationOp, ops::ExpOpMaker, exp_grad,
            ops::ActivationOpGrad);

REGISTER_OP(relu, ops::ActivationOp, ops::ReluOpMaker, relu_grad,
            ops::ActivationOpGrad);

REGISTER_OP(tanh, ops::ActivationOp, ops::TanhOpMaker, tanh_grad,
            ops::ActivationOpGrad);

REGISTER_OP(tanh_shrink, ops::ActivationOp, ops::TanhShrinkOpMaker,
            tanh_shrink_grad, ops::ActivationOpGrad);

REGISTER_OP(sqrt, ops::ActivationOp, ops::SqrtOpMaker, sqrt_grad,
            ops::ActivationOpGrad);

REGISTER_OP(abs, ops::ActivationOp, ops::AbsOpMaker, abs_grad,
            ops::ActivationOpGrad);

REGISTER_OP(reciprocal, ops::ActivationOp, ops::ReciprocalOpMaker,
            reciprocal_grad, ops::ActivationOpGrad);

REGISTER_OP(log, ops::ActivationOp, ops::LogOpMaker, log_grad,
            ops::ActivationOpGrad);

REGISTER_OP(square, ops::ActivationOp, ops::SquareOpMaker, square_grad,
            ops::ActivationOpGrad);

REGISTER_OP(softsign, ops::ActivationOp, ops::SoftsignOpMaker, softsign_grad,
            ops::ActivationOpGrad);

REGISTER_OP(brelu, ops::ActivationOp, ops::BReluOpMaker<float>, brelu_grad,
            ops::ActivationOpGrad);

REGISTER_OP(leaky_relu, ops::ActivationOp, ops::LeakyReluOpMaker<float>,
            leaky_relu_grad, ops::ActivationOpGrad);

REGISTER_OP(soft_relu, ops::ActivationOp, ops::SoftReluOpMaker<float>,
            soft_relu_grad, ops::ActivationOpGrad);

REGISTER_OP(relu6, ops::ActivationOp, ops::Relu6OpMaker<float>, relu6_grad,
            ops::ActivationOpGrad);

REGISTER_OP(pow, ops::ActivationOp, ops::PowOpMaker<float>, pow_grad,
            ops::ActivationOpGrad);

REGISTER_OP(stanh, ops::ActivationOp, ops::STanhOpMaker<float>, stanh_grad,
            ops::ActivationOpGrad);

#define REGISTER_ACTIVATION_CPU_KERNEL(act_type, functor, grad_functor)        \
  REGISTER_OP_CPU_KERNEL(                                                      \
      act_type,                                                                \
      ops::ActivationKernel<paddle::platform::CPUPlace, ops::functor<float>>); \
  REGISTER_OP_CPU_KERNEL(act_type##_grad,                                      \
                         ops::ActivationGradKernel<paddle::platform::CPUPlace, \
                                                   ops::grad_functor<float>>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_ACTIVATION_CPU_KERNEL);
