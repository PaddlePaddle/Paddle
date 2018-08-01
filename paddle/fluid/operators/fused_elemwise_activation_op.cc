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
        ctx->HasOutput("Out"),
        "Output(Out) of FusedElemwiseActivationOp op should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_GE(x_dim.size(), y_dim.size(),
                      "Rank of first input must >= rank of second input.");

    ctx->SetOutputDim("Out", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
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
    AddInput("X", "(vector<Tensor>)");
    AddInput("Y", "(vector<Tensor>)");
    AddOutput("Out", "vector<Tensor>");
    AddAttr<int>("axis",
                 "axis is used by elementwise_op, the default value is -1.")
        .SetDefault(-1);
    AddAttr<float>("scale",
                   "scale is used by scale_op, the default value is 0.0.")
        .SetDefault(0.0);
    AddAttr<bool>("recomputation", "Whether to recompute the Out.")
        .SetDefault(false);
    AddAttr<std::string>("functor_list", "The functors that should be fused.")
        .AddCustomChecker([&](const std::string &functor_list) {
          PADDLE_ENFORCE(ValidCheck(functor_list));
        });

    AddComment(R"DOC(
FusedElemwiseActivation Operator.

At present, FusedElemwiseActivation supports only two combinations of two
types(elementwise_op and activation_op) of Op.

    z = f1(x, f2(y))
    z = f1(f2(x, y))

for example:
  functor_list(f1, f2) can be represented as 'add,scale' or 'relu,add'.


)DOC");
  }

 private:
  bool ValidCheck(const std::string &functors) {
    std::unordered_set<std::string> unary_fun = {"scale", "relu"};
    std::unordered_set<std::string> binary_fun = {"elementwise_add"};

    size_t pos = functors.find(",");
    auto func_1 = functors.substr(0, pos);
    auto func_2 = functors.substr(pos + 1, functors.size());
    std::string unary_fun_str;
    if (binary_fun.count(func_1)) {
      unary_fun_str = func_2;
    } else if (binary_fun.count(func_2)) {
      unary_fun_str = func_1;
    } else {
      PADDLE_THROW("%s and %s are not included in fused_list.", func_1, func_2);
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

    std::string functor_names =
        boost::get<std::string>(op_desc_ptr->GetAttr("functor_list"));
    size_t pos = functor_names.find(",");
    PADDLE_ENFORCE_NE(pos, std::string::npos);
    auto func_1 = functor_names.substr(0, pos);
    auto func_2 = functor_names.substr(pos + 1, functor_names.size());

    op_desc_ptr->SetAttr("functor_list", func_1 + "_grad," + func_2 + "_grad");

    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

class FusedElemwiseActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
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
