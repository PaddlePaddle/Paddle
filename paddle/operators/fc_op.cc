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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

class FCOp : public NetOp {
 public:
  FCOp(const std::string &type, const framework::VariableNameMap &inputs,
       const framework::VariableNameMap &outputs,
       const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    auto x = Inputs("X");
    auto w = Inputs("W");
    auto mul_out = Outputs("MulOut");
    PADDLE_ENFORCE_EQ(
        x.size(), w.size(),
        "The size of inputs X(%d) should be the same as that of weights W(%d).",
        x.size(), w.size());
    PADDLE_ENFORCE_EQ(mul_out.size(), x.size(),
                      "The size of intermediate mul_out(%d) should be the same "
                      "as that of inputs X(%d).",
                      mul_out.size(), x.size());

    size_t n = x.size();
    PADDLE_ENFORCE_GE(n, static_cast<size_t>(1),
                      "The size of inputs X(%d) should be no less than 1.", n);

    auto x_num_col_dims = Attr<std::vector<int>>("xNumColDims");
    auto w_num_col_dims = Attr<std::vector<int>>("wNumColDims");
    PADDLE_ENFORCE_EQ(x_num_col_dims.size(), n,
                      "The size of attribute xNumColDims(%d) should be the "
                      "same as that of inputs X(%d).",
                      x_num_col_dims.size(), n);
    PADDLE_ENFORCE_EQ(w_num_col_dims.size(), n,
                      "The size of attribute wNumColDims(%d) should be the "
                      "same as that of inputs X(%d).",
                      w_num_col_dims.size(), n)

    // mul_out[i] = X[i] * W[i]
    for (size_t i = 0; i < n; i++) {
      framework::AttributeMap mul_attr;
      mul_attr["x_num_col_dims"] = static_cast<int>(x_num_col_dims[i]);
      mul_attr["y_num_col_dims"] = static_cast<int>(w_num_col_dims[i]);
      AppendOp(
          framework::OpRegistry::CreateOp("mul", {{"X", {x[i]}}, {"Y", {w[i]}}},
                                          {{"Out", {mul_out[i]}}}, mul_attr));
    }

    // sum_out = X[0] * W[0] + ... + X[n-1] * W[n-1]
    if (n > 1) {
      AppendOp(framework::OpRegistry::CreateOp(
          "sum", {{"X", {mul_out}}}, {{"Out", {Output("SumOut")}}}, {}));
    } else {
      AppendOp(framework::OpRegistry::CreateOp(
          "identity", {{"X", {mul_out[0]}}}, {{"Y", {Output("SumOut")}}}, {}));
    }

    // add_out = sum_out + b
    auto b = Input("B");
    std::string add_out = "SumOut";
    if (b != framework::kEmptyVarName) {
      add_out = "AddOut";
      AppendOp(framework::OpRegistry::CreateOp(
          "rowwise_add", {{"X", {Output("SumOut")}}, {"b", {Input("B")}}},
          {{"Out", {Output(add_out)}}}, {}));
    } else {
      if (Output("AddOut") != framework::kEmptyVarName) {
        this->Rename(Output("AddOut"), framework::kEmptyVarName);
      }
    }

    auto activation = Attr<std::string>("activation");
    AppendOp(framework::OpRegistry::CreateOp(
        activation, {{"X", {Output(add_out)}}}, {{"Y", {Output("Y")}}}, {}));
    CompleteAddOp(false);
  }
};

class FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FCOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The inputs of FC operator, a ordered vector of 2-D matrix.")
        .AsDuplicable();
    AddInput("W", "The weights of FC operator, a ordered vector of 2-D matrix.")
        .AsDuplicable();
    AddInput("B", "The 1-D bias vector of FC operator");

    AddOutput("Y", "The activated output matrix of FC operator");
    AddOutput("MulOut",
              "The intermediate outputs of FC operator, "
              "saving the product of X[i] * W[i]")
        .AsIntermediate()
        .AsDuplicable();
    AddOutput("SumOut",
              "The intermediate output of FC operator, "
              "saving the sum of products, sum(X[i] * W[i])")
        .AsIntermediate();
    AddOutput("AddOut",
              "The non-actived output of FC operator, saving X * W + b")
        .AsIntermediate();
    AddAttr<std::string>("activation", "The activation type of FC operator.")
        .SetDefault("identity")
        .InEnum({"identity", "sigmoid", "softmax"});
    AddAttr<std::vector<int>>("xNumColDims", "");
    AddAttr<std::vector<int>>("wNumColDims", "");

    AddComment(R"DOC(
Fully Connected Operator, known as Fully Connected Layer or Inner Product Layer
in Convolutional Neural Networks. Neurons in a fully connected layer have
full connections to all activations in the previous layer.
It computes an inner product of a set of
learned weights with a matrix multiplication followed by a bias offset
(optionally).

Equation:
  Y = Act(sum_n{X_i * W_i} + b)

where X_i is a 2D matrix of size (M x K), usually M is the minibatch size and
K is the number of features. W_i is also a 2D matrix of size (K x N),
where N means the number of neurons in the fully connected layer.
b is a 1D vector of size N. Thus, the output Y is a 2D matrix of size (M x N).
Activation type can be set to `identity` (default), `sigmoid` or `softmax`.

  The config api is `paddle.v2.layer.fc`.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

USE_OP(mul);
USE_OP(rowwise_add);
USE_NO_KERNEL_OP(identity);
USE_OP(sigmoid);
USE_OP(softmax);

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fc, ops::FCOp, ops::FCOpMaker);
