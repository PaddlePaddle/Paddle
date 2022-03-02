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

#include "paddle/fluid/operators/fused/fused_elemwise_activation_op.h"
#include <memory>
#include <unordered_set>

namespace paddle {
namespace operators {

bool IsUnaryCompound(const std::vector<std::string> &functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(), 2,
      platform::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(), 2));
  static std::unordered_set<std::string> binary_fun = {
      "elementwise_add", "elementwise_mul", "elementwise_add_grad",
      "elementwise_mul_grad"};
  return binary_fun.count(functor_list[1]) != 0;
}

bool HasInPlaceUnary(const std::vector<std::string> &functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(), 2,
      platform::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(), 2));
  static std::unordered_set<std::string> InplaceOpSet = {"relu", "relu_grad"};
  bool is_in_place = false;
  for (auto &func_name : functor_list) {
    is_in_place |= (InplaceOpSet.count(func_name) == 1);
  }
  return is_in_place;
}

bool InputXCanBeAbsent(const std::vector<std::string> &functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(), 2,
      platform::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(), 2));
  static std::unordered_set<std::string> binary_fun = {"elementwise_add_grad"};
  return binary_fun.count(functor_list[0]) != 0 ||
         binary_fun.count(functor_list[1]) != 0;
}

/*
 * Whether the compound function is supported.
 * For Unary(Binary(X, Y)), the intermediate_out's shape is the same the final
 * out.
 */
static bool IsSupportedCompound(const std::vector<std::string> &functors) {
  PADDLE_ENFORCE_EQ(
      functors.size(), 2UL,
      platform::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functors.size(), 2));

  static std::unordered_set<std::string> unary_fun = {"scale", "relu", "tanh",
                                                      "sigmoid", "gelu"};
  static std::unordered_set<std::string> binary_fun = {"elementwise_add",
                                                       "elementwise_mul"};

  std::string unary_fun_str;
  if (binary_fun.count(functors[0])) {
    unary_fun_str = functors[1];
  } else if (binary_fun.count(functors[1])) {
    unary_fun_str = functors[0];
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s and %s are not included in fused_list.", functors[0], functors[1]));
  }
  PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str), 1,
                    platform::errors::InvalidArgument(
                        "%s is not included in fused_list.", unary_fun_str));
  return true;
}

class FusedElemwiseActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument(
            "Input(X) of FusedElemwiseActivationOp op should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::InvalidArgument(
            "Input(Y) of FusedElemwiseActivationOp op should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of FusedElemwiseActivationOp op should not be null."));

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");

    // Whether the shape of Y is a continuous subsequence of X,
    // For more information please refer to the op's introduction.
    bool bcast_y = IsBcastY(x_dim, y_dim);

    auto &out_dim = bcast_y ? x_dim : y_dim;
    std::string out_lod = bcast_y ? "X" : "Y";

    if (ctx->Attrs().Get<bool>("save_intermediate_out")) {
      PADDLE_ENFORCE_EQ(
          ctx->HasOutput("IntermediateOut"), true,
          platform::errors::InvalidArgument(
              "Output(IntermediateOut) of FusedElemwiseActivationOp "
              "should not be null."));

      if (IsUnaryCompound(
              ctx->Attrs().Get<std::vector<std::string>>("functor_list"))) {
        // for Unary(Binary(X, Y)), the shape and lod of out and
        // intermediate_out are the same.
        ctx->SetOutputDim("IntermediateOut", out_dim);
        // set the lod of intermediate_out
        ctx->ShareLoD(out_lod, /*->*/ "IntermediateOut");
      } else {
        // for Binary(X, Unary(Y)), the shape and lod of Y and
        // intermediate_out are the same.
        ctx->SetOutputDim("IntermediateOut", y_dim);
        // set the lod of intermediate_out
        ctx->ShareLoD("Y", /*->*/ "IntermediateOut");
      }
    }
    ctx->SetOutputDim("Out", out_dim);
    ctx->ShareLoD(out_lod, /*->*/ "Out");
  }

  static bool IsBcastY(const framework::DDim &x_dim,
                       const framework::DDim &y_dim) {
    bool bcast_y = x_dim.size() >= y_dim.size();
    if (x_dim.size() == y_dim.size()) {
      for (int i = 0; i < x_dim.size(); ++i) {
        if (x_dim[i] < y_dim[i]) {
          bcast_y = false;
          break;
        }
      }
    }
    return bcast_y;
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(ctx.Input<framework::Tensor>("X")->dtype(),
                      ctx.Input<framework::Tensor>("Y")->dtype(),
                      platform::errors::InvalidArgument(
                          "The element's type of input should be the same."));
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FusedElemwiseActivationMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of fused_elemwise_activation operator.");
    AddInput(
        "Y",
        "(Tensor) The input tensor of fused_elemwise_activation operator.");
    AddOutput("Out",
              "vector<Tensor> The output tensor of fused_elemwise_activation "
              "operator.");
    AddOutput("IntermediateOut",
              "Tensor The IntermediateOut tensor of fused_elemwise_activation "
              "operator.")
        .AsIntermediate();
    AddAttr<int>("axis",
                 "axis is used by elementwise_op, the default value is -1.")
        .SetDefault(-1);
    AddAttr<float>("scale",
                   "scale is used by scale_op, the default value is 0.0.")
        .SetDefault(0.0);
    AddAttr<bool>("save_intermediate_out",
                  "Whether to save the intermediate_out.")
        .SetDefault(false);
    AddAttr<std::vector<std::string>>("functor_list",
                                      "The functors that should be fused.")
        .AddCustomChecker([&](const std::vector<std::string> &functor_list) {
          PADDLE_ENFORCE_EQ(
              IsSupportedCompound(functor_list), true,
              platform::errors::InvalidArgument(
                  "the input functors should support compounding."));
        });

    AddComment(R"DOC(
FusedElemwiseActivation Operator.

At present, FusedElemwiseActivation only supports Two kinds of compound
operators (elementwise_op and activation_op):

    Z = Binary(X, Unary(Y))
    Z = Unary(Binary(X, Y))

There are two cases for this operator:

1. The shape of $Y$ and $X$ is the same.
2. The shape of $Y$ is a continuous subsequence of $X$ or the shape of $X$ is a continuous subsequence of $Y$.

For case 2 (assume that the shape of $Y$ is a continuous subsequence of $X$ ):

1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index
   for broadcasting $Y$ onto $X$.
2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of
   subsequence, such as shape(Y) = (2, 1) => (2).

For example:

  .. code-block:: python

    shape(X) = (2, 3, 4, 5), shape(Y) = (,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0


The inputs $X$ and $Y$ can carry the different LoD information.
But the output only shares the LoD information with the one whose shape is the same with Out.
The attributions of activation_op can be get from fused_elemwise_activation_op's.
The functor_list records the functions to be fused, for example
["scale", "elementwise_add"].

)DOC");
  }
};

template <typename T>
class FusedElemwiseActivationGradMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType(this->ForwardOpType() + "_grad");

    for (auto &input_param : this->InputNames()) {
      grad_op->SetInput(input_param, this->Input(input_param));
      grad_op->SetOutput(framework::GradVarName(input_param),
                         this->InputGrad(input_param, true));
    }

    grad_op->SetInput("Out", this->Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    grad_op->SetAttrMap(this->Attrs());

    std::vector<std::string> functor_names = BOOST_GET_CONST(
        std::vector<std::string>, grad_op->GetAttr("functor_list"));

    functor_names[0] += "_grad";
    functor_names[1] += "_grad";
    grad_op->SetAttr("functor_list", functor_names);

    if (BOOST_GET_CONST(bool, grad_op->GetAttr("save_intermediate_out"))) {
      // PADDLE_ENFORCE_NE(Output("IntermediateOut").size(), 0);
      grad_op->SetInput("IntermediateOut", this->Output("IntermediateOut"));
      grad_op->SetOutput(framework::GradVarName("IntermediateOut"),
                         this->OutputGrad("IntermediateOut"));
    } else {
      grad_op->SetInput("IntermediateOut", this->EmptyOutput());
      grad_op->SetOutput(framework::GradVarName("IntermediateOut"),
                         this->EmptyOutputGrad());
    }
  }
};

class FusedElemwiseAddActivationMaker : public FusedElemwiseActivationMaker {};

template <typename T>
class FusedElemwiseAddActivationGradMaker
    : public FusedElemwiseActivationGradMaker<T> {
 public:
  using FusedElemwiseActivationGradMaker<T>::FusedElemwiseActivationGradMaker;
};

class FusedElemwiseActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@Grad) should not be null."));

    auto functor_list =
        ctx->Attrs().Get<std::vector<std::string>>("functor_list");

    if (ctx->Attrs().Get<bool>("save_intermediate_out")) {
      PADDLE_ENFORCE_EQ(ctx->HasInput("IntermediateOut"), true,
                        platform::errors::InvalidArgument(
                            "Input(IntermediateOut) should not be null."));
    } else {
      if (!InputXCanBeAbsent(functor_list)) {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::InvalidArgument("Input(X) should not be null."));
      }
    }

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    auto inter_grad_name = framework::GradVarName("IntermediateOut");

    if (ctx->HasOutput(x_grad_name)) {
      if (ctx->HasInputs("X")) {
        ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
        ctx->ShareLoD("X", x_grad_name);
      } else {
        // Currently, only when Binary is elementwise_add or elementwise_sub,
        // the "X" could be absent.
        PADDLE_ENFORCE_EQ(
            InputXCanBeAbsent(functor_list), true,
            platform::errors::InvalidArgument(
                "Only when BinaryFunctor is elementwise_add, the 'X' "
                "could be absent."));

        // Node: If "X" is absence, the shape of Y should be a continuous
        // subsequence of X, otherwise, we could not infer the shape of dx.

        ctx->SetOutputDim(x_grad_name,
                          ctx->GetInputDim(framework::GradVarName("Out")));
        ctx->ShareLoD(framework::GradVarName("Out"), x_grad_name);
      }
    }

    if (ctx->HasOutput(y_grad_name)) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("Y"), true,
          platform::errors::InvalidArgument("Input(Y) should not be null."));
      ctx->SetOutputDim(y_grad_name, ctx->GetInputDim("Y"));
      ctx->ShareLoD("Y", y_grad_name);
    }

    if (ctx->HasOutput(inter_grad_name)) {
      // For Unary(Binary(X, Y)), IntermediateOut should not be empty.
      if (IsUnaryCompound(functor_list)) {
        ctx->SetOutputDim(inter_grad_name,
                          ctx->GetInputDim(framework::GradVarName("Out")));
        ctx->ShareLoD(framework::GradVarName("Out"), inter_grad_name);
      } else {
        ctx->SetOutputDim(inter_grad_name, ctx->GetInputDim("Y"));
        ctx->ShareLoD("Y", inter_grad_name);
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

class FusedElemwiseAddActivationOp : public FusedElemwiseActivationOp {
 public:
  using FusedElemwiseActivationOp::FusedElemwiseActivationOp;
  void InferShape(framework::InferShapeContext *ctx) const override {
    FusedElemwiseActivationOp::InferShape(ctx);
    std::vector<std::string> functor_names =
        ctx->Attrs().Get<std::vector<std::string>>("functor_list");
    bool elemntwise_add_detected = false;
    for (auto names : functor_names) {
      if (names == "elementwise_add") {
        elemntwise_add_detected = true;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(
        elemntwise_add_detected, true,
        platform::errors::InvalidArgument(
            "When the FusedElemwiseAddActivationOp Is used in fused pass, the "
            "elementwise_add Op must be"
            "detected and used, Please check the fuse pass pattern"));
  }
};

class FusedElemwiseAddActivationOpGrad : public FusedElemwiseActivationOpGrad {
 public:
  using FusedElemwiseActivationOpGrad::FusedElemwiseActivationOpGrad;

  void InferShape(framework::InferShapeContext *ctx) const override {
    FusedElemwiseActivationOpGrad::InferShape(ctx);
    std::vector<std::string> functor_names =
        ctx->Attrs().Get<std::vector<std::string>>("functor_list");
    bool elemntwise_add_grad_detected = false;
    for (auto names : functor_names) {
      if (names == "elementwise_add_grad") {
        elemntwise_add_grad_detected = true;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(
        elemntwise_add_grad_detected, true,
        platform::errors::InvalidArgument(
            "When the FusedElemwiseAddActivationOpGrad Is used in fused pass, "
            "the elementwise_add_grad Op must be"
            "detected and used, Please check the fuse pass pattern"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(
    FusedElemwiseAddActivationNoNeddBufVarInferer, "X", "Y");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_elemwise_activation, ops::FusedElemwiseActivationOp,
    ops::FusedElemwiseActivationMaker,
    ops::FusedElemwiseActivationGradMaker<paddle::framework::OpDesc>,
    ops::FusedElemwiseActivationGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_elemwise_activation_grad,
                  ops::FusedElemwiseActivationOpGrad);

REGISTER_OP_CPU_KERNEL(
    fused_elemwise_activation,
    ops::FusedElemwiseActivationKernel<paddle::platform::CPUDeviceContext,
                                       float>,
    ops::FusedElemwiseActivationKernel<paddle::platform::CPUDeviceContext,
                                       double>);

REGISTER_OP_CPU_KERNEL(
    fused_elemwise_activation_grad,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CPUDeviceContext,
                                           float>,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CPUDeviceContext,
                                           double>);

// for memory optimization, we register the fused_elemwise_add_activation OP
REGISTER_OPERATOR(
    fused_elemwise_add_activation, ops::FusedElemwiseAddActivationOp,
    ops::FusedElemwiseAddActivationMaker,
    ops::FusedElemwiseAddActivationGradMaker<paddle::framework::OpDesc>,
    ops::FusedElemwiseAddActivationGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_elemwise_add_activation_grad,
                  ops::FusedElemwiseAddActivationNoNeddBufVarInferer,
                  ops::FusedElemwiseAddActivationOpGrad);

REGISTER_OP_CPU_KERNEL(
    fused_elemwise_add_activation,
    ops::FusedElemwiseActivationKernel<paddle::platform::CPUDeviceContext,
                                       float>,
    ops::FusedElemwiseActivationKernel<paddle::platform::CPUDeviceContext,
                                       double>);

REGISTER_OP_CPU_KERNEL(
    fused_elemwise_add_activation_grad,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CPUDeviceContext,
                                           float>,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CPUDeviceContext,
                                           double>);
