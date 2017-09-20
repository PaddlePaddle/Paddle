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
    PADDLE_ENFORCE(!Inputs("X").empty(),
                   "Inputs(X) of FCOp should not be null.");
    PADDLE_ENFORCE(!Inputs("W").empty(),
                   "Inputs(W) of FCOp should not be null.");
    PADDLE_ENFORCE(!Outputs("MulOut").empty(),
                   "Outputs(MulOut) of FCOp should not be null.");
    PADDLE_ENFORCE_NE(Output("Out"), framework::kEmptyVarName,
                      "Output(Out) of FCOp should not be null.");

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

    // Set all values or set no values (use the default value)
    if (!x_num_col_dims.empty()) {
      PADDLE_ENFORCE_EQ(x_num_col_dims.size(), n,
                        "The size of attribute xNumColDims(%d) should be the "
                        "same as that of inputs X(%d).",
                        x_num_col_dims.size(), n);
    } else {
      x_num_col_dims.resize(n);
      for (size_t i = 0; i < n; i++) {
        x_num_col_dims[i] = 1;
      }
    }

    // mul_out[i] = X[i] * W[i]
    for (size_t i = 0; i < n; i++) {
      framework::AttributeMap mul_attr;
      mul_attr["x_num_col_dims"] = static_cast<int>(x_num_col_dims[i]);
      mul_attr["y_num_col_dims"] = static_cast<int>(1);
      AppendOp(
          framework::OpRegistry::CreateOp("mul", {{"X", {x[i]}}, {"Y", {w[i]}}},
                                          {{"Out", {mul_out[i]}}}, mul_attr));
    }

    // sum_out = X[0] * W[0] + ... + X[n-1] * W[n-1]
    auto sum_out = mul_out[0];
    if (n > 1) {
      PADDLE_ENFORCE_NE(Output("SumOut"), framework::kEmptyVarName,
                        "Output(SumOut) of FCOp should not be null when the "
                        "size of Inputs(X) > 1.");

      sum_out = Output("SumOut");
      AppendOp(framework::OpRegistry::CreateOp("sum", {{"X", {mul_out}}},
                                               {{"Out", {sum_out}}}, {}));
    } else {
      if (Output("SumOut") != framework::kEmptyVarName) {
        this->Rename(Output("SumOut"), framework::kEmptyVarName);
      }
    }

    // add_out = sum_out + b
    auto b = Input("B");
    auto add_out = sum_out;
    if (b != framework::kEmptyVarName) {
      PADDLE_ENFORCE_NE(
          Output("AddOut"), framework::kEmptyVarName,
          "Output(AddOut) of FCOp should not be null when Input(B) is set.");

      add_out = Output("AddOut");
      AppendOp(framework::OpRegistry::CreateOp(
          "rowwise_add", {{"X", {sum_out}}, {"b", {Input("B")}}},
          {{"Out", {add_out}}}, {}));
    } else {
      if (Output("AddOut") != framework::kEmptyVarName) {
        this->Rename(Output("AddOut"), framework::kEmptyVarName);
      }
    }

    auto activation = Attr<std::string>("activation");
    AppendOp(framework::OpRegistry::CreateOp(activation, {{"X", {add_out}}},
                                             {{"Y", {Output("Out")}}}, {}));
    CompleteAddOp(false);
  }
};

class FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FCOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(A vector of Tensors) each input Tensor can be of arbitrary "
             "dimension, and will be reshaped to a 2-D matrix of size "
             "(minibatch, number_of_input_features) according to attribute "
             "xNumColDims.")
        .AsDuplicable();
    AddInput("W",
             "(A vector of Tensors) the weights of FC operator, a "
             "vector of 2-D matrix of size "
             "(number_of_input_features, number_of_neurons).")
        .AsDuplicable();
    AddInput("B",
             "(Tensor) the bias of FC operator, a 1-D vector of size "
             "number_of_neurons.");

    AddOutput("Out",
              "(Tensor) the activated output matrix of FC operator, a 2-D "
              "matrix of size (minibatch, number_of_neurons).");
    AddOutput("MulOut",
              "(A vector of Tensors) the intermediate outputs of FC operator, "
              "each Tensor saving the product of X_i * W_i.")
        .AsIntermediate()
        .AsDuplicable();
    AddOutput(
        "SumOut",
        "(Tensor) the intermediate output of FC operator, "
        "saving the sum of the products of X and W, that is sum{X_i * W_i}.")
        .AsIntermediate();
    AddOutput("AddOut",
              "(Tensor) the non-actived output of FC operator, "
              "saving sum{X_i * W_i} + B.")
        .AsIntermediate();
    AddAttr<std::string>(
        "activation",
        "(string, default identity) the activation type of FC operator.")
        .SetDefault("identity")
        .InEnum({"identity", "sigmoid", "softmax"});
    AddAttr<std::vector<int>>(
        "xNumColDims",
        "(std::vector<int>) The inputs Tensors of FC operator can be of "
        "more than 2 dimensions. In that case, each input Tensor `X_i` will be "
        "reshaped to a 2-D matrix. The matrix's first dimension "
        "(the length of column) will be the product of `X_i`'s last "
        "`xNumColDims_i` dimensions, that is "
        "`X_i.dims[0] x ... x X_i.dims[xNumColDims_i - 1]`. "
        "The matrix's second dimension (the length of row) will be the product "
        "of `X_i`'s first `rank - xNumColDims_i` dimensions, that is "
        "`X_i.dims[xNumColDims_i] x ... x X_i.dims[rank - 1]`)")
        .SetDefault(std::vector<int>{});

    AddComment(R"DOC(
Fully Connected Operator, known as Fully Connected Layer or Inner Product Layer
in Convolutional Neural Networks. Neurons in a fully connected layer have
full connections to all activations in the previous layer.
It computes an inner product of a set of
learned weights with a matrix multiplication followed by a bias offset
(optionally).

Equation:
  Out = Act(sum_n{X_i * W_i} + B)

where X_i is Tensor that will be reshaped to a 2-D matrix of size (M x K),
usually M is the minibatch size and K is the number of input features.
W_i is a 2-D matrix of size (K x N), where N means the number of neurons
in the fully connected layer. B is a 1-D vector of size N.
Thus, the output Out is a 2-D matrix of size (M x N).
Activation type can be set to `identity` (default), `sigmoid` or `softmax`.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fc, ops::FCOp, ops::FCOpMaker);
