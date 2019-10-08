/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

class MultiHeadMatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInput("Q"), true,
                      "Input(Q) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("K"), true,
                      "Input(K) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("V"), true,
                      "Input(V) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasQ"), true,
                      "Input(BiasQ) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasK"), true,
                      "Input(BiasQ) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasV"), true,
                      "Input(BiasQ) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasQK"), true,
                      "Input(BiasQK) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasOutput("Out"), true,
                      "Output(Out) of MatMulOp should not be null.");

    auto dim_q = context->GetInputDim("Q");
    PADDLE_ENFORCE_GT(dim_q.size(), 2,
                      "Multihead input should be at least 3-D tensor.");

    auto dim_k = context->GetInputDim("K");
    PADDLE_ENFORCE_GT(dim_q.size(), 2,
                      "Multihead input should be at least 3-D tensor.");

    auto dim_v = context->GetInputDim("V");
    PADDLE_ENFORCE_GT(dim_q.size(), 2,
                      "Multihead input should be at least 3-D tensor.");

    PADDLE_ENFORCE_EQ(dim_q[0], dim_k[0],
                      "Multihead input should have same batch size");
    PADDLE_ENFORCE_EQ(dim_q[0], dim_v[0],
                      "Multihead input should have same batch size");

    PADDLE_ENFORCE_EQ(dim_q[1], dim_k[1],
                      "Multihead input should have same size");
    PADDLE_ENFORCE_EQ(dim_q[1], dim_v[1],
                      "Multihead input should have same size");

    PADDLE_ENFORCE_EQ(dim_q[2], dim_k[2],
                      "Multihead input should have same size");
    PADDLE_ENFORCE_EQ(dim_q[2], dim_v[2],
                      "Multihead input should have same size");

    auto dim_bias_q = context->GetInputDim("BiasQ");
    PADDLE_ENFORCE_GT(dim_bias_q.size(), 0,
                      "Multihead input should be at least 1-D tensor.");
    auto dim_bias_k = context->GetInputDim("BiasK");
    PADDLE_ENFORCE_GT(dim_bias_k.size(), 0,
                      "Multihead input should be at least 1-D tensor.");
    auto dim_bias_v = context->GetInputDim("BiasV");
    PADDLE_ENFORCE_GT(dim_bias_v.size(), 0,
                      "Multihead input should be at least 1-D tensor.");

    PADDLE_ENFORCE_EQ(dim_bias_q[0], dim_bias_k[0],
                      "Multihead input bias should have same batch size");
    PADDLE_ENFORCE_EQ(dim_bias_q[0], dim_bias_v[0],
                      "Multihead input bias should have same batch size");

    PADDLE_ENFORCE_EQ(dim_bias_q[1], dim_bias_k[1],
                      "Multihead input bias should have same size");
    PADDLE_ENFORCE_EQ(dim_bias_q[1], dim_bias_v[1],
                      "Multihead input bias should have same size");

    auto dim_bias_qk = context->GetInputDim("BiasQK");
    PADDLE_ENFORCE_GT(dim_bias_qk.size(), 3,
                      "Multihead input bias qk should be at least 4-D tensor.");

    int head_number = context->Attrs().Get<int>("head_number");
    PADDLE_ENFORCE_GT(head_number, 1,
                      "Multihead input head number should be at least 1.");

    context->SetOutputDim("Out", dim_q);
    context->ShareLoD("Q", /*->*/ "Out");
  }
};

class MultiHeadMatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "The first input of MultiHeadMatMul op");
    AddInput("K", "The second input of MMultiHeadMatMul op");
    AddInput("V", "The third input of MultiHeadMatMul op");
    AddInput("BiasQ", "The first bias input of MultiHeadMatMul op");
    AddInput("BiasK", "The second bias input of MultiHeadMatMul op");
    AddInput("BiasV", "The third  bias input of MultiHeadMatMul op");
    AddInput("BiasQK", "The QK bias input of MultiHeadMatMul op");
    AddOutput("Out", "The output of MultiHeadMatMul op");
    AddAttr<bool>("transpose_Q",
                  R"DOC(If true, use the transpose of `Q`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_K",
                  R"DOC(If true, use the transpose of `K`.
        )DOC")
        .SetDefault(true);
    AddAttr<bool>("transpose_V",
                  R"DOC(If true, use the transpose of `V`.
        )DOC")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(1);
    AddComment(R"DOC(
MultiHeadMatMul Operator.

This op is used for optimize multi head calculation in ernie model.
Not suggest to use in other case except has same structure as ernie.

Example of matrix multiplication with head_number of H
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]

Both the input `Q` and `K` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input `Q`, because
they are the same.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(multihead_matmul, ops::MultiHeadMatMulOp,
                             ops::MultiHeadMatMulOpMaker);
