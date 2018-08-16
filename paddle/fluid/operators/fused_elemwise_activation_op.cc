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

#include <string>
#include <vector>

#include "paddle/fluid/operators/fused_elemwise_activation_op.h"

namespace paddle {
namespace operators {

class FusedElemwiseActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of FusedElemwiseActivationOp op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Y"),
        "Input(Y) of FusedElemwiseActivationOp op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutputs("Out"),
        "Output(Out) of FusedElemwiseActivationOp op should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");

    // TODO(zcd): whether Y can broadcast to X or X can broadcast to Y
    bool bcast_y = x_dim.size() >= y_dim.size();
    if (x_dim.size() == y_dim.size()) {
      for (int i = 0; i < x_dim.size(); ++i) {
        if (x_dim[i] < y_dim[i]) {
          bcast_y = false;
          break;
        }
      }
    }

    auto &out_dim = bcast_y ? x_dim : y_dim;

    if (ctx->Attrs().Get<bool>("keep_intermediate_value")) {
      PADDLE_ENFORCE_EQ(ctx->Outputs("Out").size(), 2,
                        "The option of 'keep_intermediate_value' is opened, "
                        "so the number of 'Out' should be two.");
      // TODO(zcd): hard code.
      if (ctx->Attrs().Get<std::vector<std::string>>("functor_list")[1] ==
          "elementwise_add") {
        ctx->SetOutputsDim("Out", {out_dim, out_dim});
      } else {
        ctx->SetOutputsDim("Out", {out_dim, y_dim});
      }
    } else {
      PADDLE_ENFORCE_EQ(
          ctx->Outputs("Out").size(), 1,
          "The option of 'keep_intermediate_value' is not opened, "
          "so the number of 'Out' should be only one.");

      ctx->SetOutputsDim("Out", {out_dim});
      ctx->ShareLoD("X", /*->*/ "Out");  // Is it necessary?
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(ctx.Input<framework::Tensor>("X")->type(),
                      ctx.Input<framework::Tensor>("Y")->type(),
                      "The element's type of input should be the same.");
    auto input_data_type =
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type());
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
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
              "operator."
              "If the option of 'keep_intermediate_value' is opened, the "
              "number of 'Out' should be two."
              "else, the number of 'Out' should be one.")
        .AsDuplicable();
    AddAttr<int>("axis",
                 "axis is used by elementwise_op, the default value is -1.")
        .SetDefault(-1);
    AddAttr<float>("scale",
                   "scale is used by scale_op, the default value is 0.0.")
        .SetDefault(0.0);
    AddAttr<bool>(
        "recomputation",
        "Whether to recompute the Out."
        "fused_elemwise_activation_grad has two methods to get the "
        "dx and dy, one is to use the 'Out', and the other is not to use it. "
        "The former method will save the time of recomputing the "
        "'Out', but it must occupy the memory to store the 'out'. "
        "While, the later method can avoid occupying the memory, "
        "but it must recompute the 'Out'. The default value is true.")
        .SetDefault(true);
    AddAttr<bool>("keep_intermediate_value",
                  "scale is used by scale_op, the default value is 0.0.")
        .SetDefault(false);
    AddAttr<std::vector<std::string>>("functor_list",
                                      "The functors that should be fused.")
        .AddCustomChecker([&](const std::vector<std::string> &functor_list) {
          PADDLE_ENFORCE(ValidCheck(functor_list));
        });

    AddComment(R"DOC(
FusedElemwiseActivation Operator.

At present, FusedElemwiseActivation only supports Two kinds of compound
operators (elementwise_op and activation_op):

    Z = Binary(X, Unary(Y))
    Z = Unary(Binary(X, Y))

The input 'X' and 'Y' can be summed by broadcasting 'X' or 'Y'.
The attributions of activation_op can be get from fused_elemwise_activation_op's
attributions. functor_list records the functors to be fused, for example
"scale,elementwise_add".

)DOC");
  }

 private:
  bool ValidCheck(const std::vector<std::string> &functors) {
    std::unordered_set<std::string> unary_fun = {"scale", "relu"};
    std::unordered_set<std::string> binary_fun = {"elementwise_add"};

    std::string unary_fun_str;
    if (binary_fun.count(functors[0])) {
      unary_fun_str = functors[1];
    } else if (binary_fun.count(functors[1])) {
      unary_fun_str = functors[0];
    } else {
      PADDLE_THROW("%s and %s are not included in fused_list.", functors[0],
                   functors[1]);
    }
    PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str), 1,
                      "%s is not included in fused_list.", unary_fun_str);
    return true;
  }
};

class FusedElemwiseActivationGradMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *op_desc_ptr = new framework::OpDesc();
    op_desc_ptr->SetType(this->ForwardOpType() + "_grad");

    for (auto &input_param : this->InputNames()) {
      op_desc_ptr->SetInput(input_param, this->Input(input_param));
      op_desc_ptr->SetOutput(framework::GradVarName(input_param),
                             this->InputGrad(input_param, true));
    }

    for (auto &output_param : this->OutputNames()) {
      op_desc_ptr->SetInput(output_param, this->Output(output_param));
      op_desc_ptr->SetInput(framework::GradVarName(output_param),
                            this->OutputGrad(output_param));
    }

    op_desc_ptr->SetAttrMap(this->Attrs());

    std::vector<std::string> functor_names =
        boost::get<std::vector<std::string>>(
            op_desc_ptr->GetAttr("functor_list"));
    functor_names[0] += "_grad";
    functor_names[1] += "_grad";
    op_desc_ptr->SetAttr("functor_list", functor_names);
    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

class FusedElemwiseActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->Attrs().Get<bool>("keep_intermediate_value")) {
      PADDLE_ENFORCE_EQ(ctx->Inputs(framework::GradVarName("Out")).size(), 2);
    } else {
      PADDLE_ENFORCE_EQ(ctx->Inputs(framework::GradVarName("Out")).size(), 1);
    }

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      PADDLE_ENFORCE(ctx->HasInputs(framework::GradVarName("Out")),
                     "Input(Out@GRAD) should not be null");
      auto out_dims = ctx->GetInputsDim(framework::GradVarName("Out"));
      ctx->SetOutputDim(x_grad_name, out_dims[0]);
    }
    if (ctx->HasOutput(y_grad_name)) {
      PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
      ctx->SetOutputDim(y_grad_name, ctx->GetInputDim("Y"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type_index = ctx.Input<framework::Tensor>("X")->type();
    PADDLE_ENFORCE_EQ(input_data_type_index,
                      ctx.Input<framework::Tensor>("Y")->type(),
                      "The element's type of input should be the same.");
    PADDLE_ENFORCE_EQ(
        input_data_type_index,
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type(),
        "The element's type of input should be the same.");

    auto input_data_type = framework::ToDataType(input_data_type_index);
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_elemwise_activation, ops::FusedElemwiseActivationOp,
                  ops::FusedElemwiseActivationMaker,
                  ops::FusedElemwiseActivationGradMaker);
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
