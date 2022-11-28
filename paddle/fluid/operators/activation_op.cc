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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/phi/backends/dynload/port.h"
#include "paddle/phi/infermeta/backward.h"

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

template <typename GradFunctor>
static constexpr bool CanInplaceAct() {
  return GradFunctor::FwdDeps() == ActBwdOpFwdDeps::kDepOut ||
         GradFunctor::FwdDeps() == ActBwdOpFwdDeps::kNoDeps;
}

#define REGISTER_ACTIVATION_OP_MAKER(OP_NAME, OP_COMMENT)           \
  class OP_NAME##OpMaker                                            \
      : public ::paddle::framework::OpProtoAndCheckerMaker {        \
   public:                                                          \
    void Make() override {                                          \
      AddInput("X",                                                 \
               "Input of " #OP_NAME                                 \
               " operator, an N-D Tensor, with data type float32, " \
               "float64 or float16.");                              \
      AddOutput("Out",                                              \
                "Output of " #OP_NAME                               \
                " operator, a Tensor with shape same as input.");   \
      AddComment(OP_COMMENT);                                       \
    }                                                               \
  }

template <ActBwdOpFwdDeps kDepValue, typename T>
class ActivationGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());

    if ((static_cast<int>(kDepValue) &
         static_cast<int>(ActBwdOpFwdDeps::kDepX)) ||
        FLAGS_use_mkldnn ||
        (op->HasAttr("use_mkldnn") &&
         PADDLE_GET_CONST(bool, op->GetAttr("use_mkldnn")))) {
      op->SetInput("X", this->Input("X"));  // x
    }

    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      op->SetInput("Out", this->Output("Out"));  // out
    }
  }
};

framework::OpKernelType GetKernelType(const framework::ExecutionContext& ctx,
                                      const framework::OperatorWithKernel& oper,
                                      const std::string& name) {
  auto data_type = oper.IndicateVarDataType(ctx, name);
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
  return framework::OpKernelType(data_type, ctx.GetPlace());
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
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
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

class BReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input is a multi-dimensional Tensor. The data type is "
             "float32, float64.");
    AddOutput("Out",
              "The output is a multi-dimensional Tensor which has same "
              "dimension and data type as the ``X``.");
    AddAttr<float>("t_min", "The min marginal value of BRelu")
        .SetDefault(static_cast<float>(0));
    AddAttr<float>("t_max", "The max marginal value of BRelu")
        .SetDefault(static_cast<float>(24));
    AddComment(R"DOC(
BRelu Activation Operator.

$$out = \min(\max(x, t_{min}), t_{max})$$

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

$$out = \ln(1 + \exp(\max(\min(x, threshold), -threshold)))$$

)DOC");
  }
};

class Relu6OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Input of relu6 operator, an N-D Tensor, "
             "with data type float32, float64.");
    AddOutput(
        "Out",
        "Output of relu6 operator, a Tensor with the same shape as input.");
    AddAttr<float>("threshold",
                   "The threshold value of Relu6. Default is 6.0. ")
        .SetDefault(6.0f);
    AddComment(R"DOC(
Relu6 Activation Operator.

$$out = \min(\max(0, x), threshold)$$

)DOC");
  }
};

class PowOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Pow operator");
    AddInput("FactorTensor",
             "(Tensor<float>, optional). If provided, pow will use this"
             "The shape of FactorTensor MUST BE [1]."
             "it has higher priority than attr(factor).")
        .AsDispensable();
    AddOutput("Out", "Output of Pow operator");
    AddAttr<float>("factor", "The exponential factor of Pow").SetDefault(1.0f);
    AddComment(R"DOC(
Pow Activation Operator.

$$out = x^{factor}$$

)DOC");
  }
};

class STanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Input of STanh operator."
             " A Tensor with type float32, float64.");
    AddOutput("Out", "Output of STanh operator. A Tensor with type float32.");
    AddAttr<float>("scale_a", "The scale parameter of a for the input. ")
        .SetDefault(0.67f);
    AddAttr<float>("scale_b", "The scale parameter of b for the input")
        .SetDefault(1.7159f);
    AddComment(R"DOC(
STanh Activation Operator.

$$out = b * \\frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}$$

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

$$out = \\frac{x}{1 + e^{- \beta \ x}}$$

)DOC");
  }
};

class MishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Mish operator");
    AddOutput("Out", "Output of Mish operator");
    AddAttr<float>(
        "threshold",
        "Constant threshold of softplus in Mish operator. Approximate value "
        "of softplus will be used if absolute value of input is greater than "
        ":attr:`threshold`")
        .SetDefault(20.f);
    AddComment(R"DOC(
Mish Activation Operator.

..  math::
    softplus(x) = \begin{cases}
            x, \text{if } x > \text{threshold} \\
            \ln(1 + e^{x}),  \text{otherwise}
          \end{cases}

    out = x * \tanh(softplus(x))

)DOC");
  }
};

class HardSwishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of HardSwish operator");
    AddOutput("Out", "Output of HardSwish operator");
    AddAttr<float>("threshold", "The threshold parameter of HardSwish operator")
        .SetDefault(6.0f);
    AddAttr<float>("scale", "The scale parameter of HardSwish operator")
        .SetDefault(6.0f);
    AddAttr<float>("offset", "The offset parameter of HardSwish operator")
        .SetDefault(3.0f);
    AddComment(R"DOC(
HardSwish Activation Operator.

The hard version of swish(https://arxiv.org/pdf/1905.02244.pdf).

$$out = \frac{x * (min(max(0, x+offset), threshold))}{scale}$$

The threshold and scale should be positive. The offset can be either positive or negative.
The default parameters are set according to the above reference.
It is recommended to use the defaults for this activation.

)DOC");
  }
};

template <ActBwdOpFwdDeps kDepValue>
class ActivationOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
      if (ctx->HasOutput("DX")) {
        ctx->ShareDim("X", "DX");
        ctx->ShareLoD("X", "DX");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("X", "DDOut");
        ctx->ShareLoD("X", "DDOut");
      }
    }
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      if (ctx->HasOutput("DOut")) {
        ctx->ShareDim("Out", "DOut");
        ctx->ShareLoD("Out", "DOut");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("Out", "DDOut");
        ctx->ShareLoD("Out", "DDOut");
      }
      if (ctx->HasOutput("DOutNew")) {
        ctx->ShareDim("Out", "DOutNew");
        ctx->ShareLoD("Out", "DOutNew");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "DDX");
  }
};

template <ActBwdOpFwdDeps kDepValue>
class ActivationOpDoubleGrad2 : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("X", "DDOut");
        ctx->ShareLoD("X", "DDOut");
      }
    }
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("Out", "DDOut");
        ctx->ShareLoD("Out", "DDOut");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "DDX");
  }
};

template <ActBwdOpFwdDeps kDepValue>
class ActivationOpTripleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
      if (ctx->HasOutput("DX")) {
        ctx->ShareDim("X", "DX");
        ctx->ShareLoD("X", "DX");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("X", "DDOut");
        ctx->ShareLoD("X", "DDOut");
      }
    }
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      if (ctx->HasOutput("D_DOut")) {
        ctx->ShareDim("Out", "D_DOut");
        ctx->ShareLoD("Out", "D_DOut");
      }
      if (ctx->HasOutput("D_OutNew")) {
        ctx->ShareDim("Out", "D_OutNew");
        ctx->ShareLoD("Out", "D_OutNew");
      }
      if (ctx->HasOutput("D_DDx")) {
        ctx->ShareDim("DDX", "D_DDx");
        ctx->ShareLoD("DDX", "D_DDx");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "DDX");
  }
};

DECLARE_INPLACE_OP_INFERER(ActivationGradOpInplaceInferer,
                           {framework::GradVarName("Out"),  // dout
                            framework::GradVarName("X")});  // dx
DECLARE_INPLACE_OP_INFERER(ActivationDoubleGradOpInplaceInferer,
                           {"DDX", "DDOut"});
DECLARE_INPLACE_OP_INFERER(ActivationTripleGradOpInplaceInferer,
                           {"DDX", "D_DOut"});

template <typename T>
class PowGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("pow_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework ::GradVarName("X"), this->InputGrad("X"));
    op->SetInput("FactorTensor", this->Input("FactorTensor"));
    op->SetAttrMap(this->Attrs());
  }
};
template <typename T>
class PowDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("pow_double_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework ::GradVarName("X")));
    op->SetOutput("DX", this->InputGrad("X"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    op->SetInput("FactorTensor", this->Input("FactorTensor"));
    op->SetAttrMap(this->Attrs());
  }
};
template <typename T>
class PowTripleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("pow_triple_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("DOut", this->Input("DOut"));
    op->SetInput("DDX", this->Input("DDX"));
    op->SetInput("D_DX", this->OutputGrad("DX"));
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));
    op->SetOutput("D_X", this->InputGrad("X"));
    op->SetOutput("D_DOut", this->InputGrad("DOut"));
    op->SetOutput("D_DDX", this->InputGrad("DDX"));
    op->SetInput("FactorTensor", this->Input("FactorTensor"));
    op->SetAttrMap(this->Attrs());
  }
};
class PowOp : public framework::OperatorWithKernel {
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

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "FactorTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class PowOpGrad : public framework::OperatorWithKernel {
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

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "FactorTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class PowOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "X");
  }
};

class PowOpTripleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "X");
  }
};
DECLARE_INPLACE_OP_INFERER(ActFwdInplaceInferer, {"X", "Out"});
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_OP(KERNEL_TYPE, OP_NAME, functor, grad_functor) \
  REGISTER_OPERATOR(                                                        \
      KERNEL_TYPE,                                                          \
      ops::ActivationOp,                                                    \
      ops::OP_NAME##OpMaker,                                                \
      ops::ActivationOpInferVarType,                                        \
      ops::ActivationGradOpMaker<ops::grad_functor<float>::FwdDeps(),       \
                                 paddle::framework::OpDesc>,                \
      ops::ActivationGradOpMaker<ops::grad_functor<float>::FwdDeps(),       \
                                 paddle::imperative::OpBase>,               \
      std::conditional<ops::CanInplaceAct<ops::grad_functor<float>>(),      \
                       ops::ActFwdInplaceInferer,                           \
                       void>::type);                                        \
  REGISTER_OPERATOR(KERNEL_TYPE##_grad,                                     \
                    ops::ActivationOpGrad,                                  \
                    ops::ActivationGradOpInplaceInferer);

#define REGISTER_ACTIVATION_CPU_KERNEL(                                     \
    act_type, op_name, functor, grad_functor)                               \
  REGISTER_OP_CPU_KERNEL(                                                   \
      act_type,                                                             \
      ops::ActivationKernel<phi::CPUContext, ops::functor<float>>,          \
      ops::ActivationKernel<phi::CPUContext, ops::functor<double>>);        \
  REGISTER_OP_CPU_KERNEL(                                                   \
      act_type##_grad,                                                      \
      ops::ActivationGradKernel<phi::CPUContext, ops::grad_functor<float>>, \
      ops::ActivationGradKernel<phi::CPUContext, ops::grad_functor<double>>);

FOR_EACH_ACTIVATION_OP(REGISTER_ACTIVATION_OP);
FOR_EACH_ACTIVATION_OP(REGISTER_ACTIVATION_CPU_KERNEL);

REGISTER_ACTIVATION_OP(brelu, BRelu, BReluFunctor, BReluGradFunctor);
REGISTER_ACTIVATION_OP(relu6, Relu6, Relu6Functor, Relu6GradFunctor);
REGISTER_ACTIVATION_OP(mish, Mish, MishFunctor, MishGradFunctor);
REGISTER_ACTIVATION_OP(stanh, STanh, STanhFunctor, STanhGradFunctor);
REGISTER_ACTIVATION_OP(hard_swish,
                       HardSwish,
                       HardSwishFunctor,
                       HardSwishGradFunctor);
REGISTER_ACTIVATION_OP(swish, Swish, SwishFunctor, SwishGradFunctor);

/* ==========================   pow register  ============================ */
DECLARE_INFER_SHAPE_FUNCTOR(pow_double_grad,
                            PowDoubleGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralBinaryGradInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(pow_triple_grad,
                            PowTripleGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralTernaryGradInferMeta));

REGISTER_OPERATOR(
    pow,
    ops::PowOp,
    ops::PowOpMaker,
    ops::ActivationOpInferVarType,
    ops::PowGradOpMaker<paddle::framework::OpDesc>,
    ops::PowGradOpMaker<paddle::imperative::OpBase>,
    std::conditional<ops::CanInplaceAct<ops::PowGradFunctor<float>>(),
                     ops::ActFwdInplaceInferer,
                     void>::type);
REGISTER_OPERATOR(pow_grad,
                  ops::PowOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::PowDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::PowDoubleGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(pow_double_grad,
                  ops::PowOpDoubleGrad,
                  ops::ActivationDoubleGradOpInplaceInferer,
                  ops::PowTripleGradOpMaker<paddle::framework::OpDesc>,
                  ops::PowTripleGradOpMaker<paddle::imperative::OpBase>,
                  PowDoubleGradInferShapeFunctor);
REGISTER_OPERATOR(pow_triple_grad,
                  ops::PowOpTripleGrad,
                  PowTripleGradInferShapeFunctor);
/* ========================================================================== */

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(leaky_relu)
    .AddCheckpoint(
        R"ROC(fix leaky_relu, bahavior changed when alpha < 0 or alpha > 1)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .BugfixWithBehaviorChanged(
                "leaky_relu calculate formula before checkponit: out = max(x, "
                "alpha * x); after checkpoint: out = x if x > 0 else alpha * "
                "x"));

REGISTER_OP_VERSION(hard_shrink)
    .AddCheckpoint(
        R"ROC(fix hard_shrink, bahavior changed when threshold<0)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .BugfixWithBehaviorChanged(
                "hard_shrink calculate formula before checkponit: out = x * "
                "((x < -threshold) + (x > threshold)); after checkpoint: out = "
                "x * (((x < -threshold) + (x > threshold)) > 0)"));

REGISTER_OP_VERSION(softplus).AddCheckpoint(
    R"ROC(add new attributes [beta] and [threshold], and the formula is changed to "
         " softplus(x) = \\frac{1}{beta} * \\log(1 + e^{beta * x}) \\\\ \\text{For numerical"
         " stability, the implementation reverts to the linear function when: beta * x > threshold.})ROC",
    paddle::framework::compatible::OpVersionDesc()
        .NewAttr("beta", "The beta value of the new formula", 1.0f)
        .NewAttr("threshold", "The threshold value of the new formula", 20.0f));

REGISTER_OP_VERSION(mish).AddCheckpoint(
    R"ROC(add new attributes [use_mkldnn], and when computing softplus the formula is changed as the new veriosn of softplus)ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "use_mkldnn",
        "(bool, default false) Only used in mkldnn kernel",
        false));

/* ========================================================================== */
