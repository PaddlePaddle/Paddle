/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class SparseAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Q",
        "(Tensor), The input tensor of query in attention, "
        "whose dimension : `[batch_size, num_heads, target_len, head_dim]`.");
    AddInput(
        "K",
        "(Tensor), The input tensor of key in attention, "
        "whose dimension : `[batch_size, num_heads, target_len, head_dim]`.");
    AddInput(
        "V",
        "(Tensor), The input tensor of value in attention, "
        "whose dimension : `[batch_size, num_heads, target_len, head_dim]`.");
    AddInput("Offset",
             "(Tensor, default: Tensor<int32>), The input tensor of offset in "
             "CSR sparse format, "
             "whose dimension : `[batch_size, num_heads, target_len + 1]`.");
    AddInput("Columns",
             "(Tensor, default: Tensor<int32>), The input tensor of columns in "
             "CSR sparse format, "
             "whose dimension : `[batch_size, num_heads, sparse_nnz_num]`.");
    AddInput("KeyPaddingMask",
             "(Tensor), The input tensor of key padding mask"
             "whose dimension : `[batch_size, target_len]`.")
        .AsDispensable();
    AddInput("AttnMask",
             "(Tensor), The input tensor of attention mask"
             "whose dimension : `[target_len, target_len]`.")
        .AsDispensable();
    AddOutput(
        "Out",
        "(Tensor), The output tensor of result in attention, "
        "whose dimension : `[batch_size, num_heads, target_len, head_dim]`.");
    AddOutput("SparseDotSdd",
              "(Tensor), The output tensor of result in SparseDotSdd step, "
              "whose dimension : `[batch_size, num_heads, sparse_nnz_dim]`.")
        .AsIntermediate();
    AddOutput("Softmax",
              "(Tensor), The output tensor of result in Softmax step, "
              "whose dimension : `[batch_size, num_heads, sparse_nnz_dim]`.")
        .AsIntermediate();
    AddComment(R"DOC(
      Compute the value of the sparse attention module. Its input value includes five tensors.
      Q, K, and V represent query, key, and value in the Attention module, respectively. 
      The CSR format is used to represent the sparsity feature in the Attention module. 
      The CSR format contains two tensors, offset and columns.
      )DOC");
  }
};

class SparseAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Q"), "Input", "Q", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("K"), "Input", "K", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("V"), "Input", "V", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("Offset"), "Input", "Offset",
                   "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("Columns"), "Input", "Columns",
                   "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("SparseDotSdd"), "Output", "SparseDotSdd",
                   "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("Softmax"), "Output", "Softmax",
                   "sparse_attention");

    auto dims_q = ctx->GetInputDim("Q");
    auto dims_k = ctx->GetInputDim("K");
    auto dims_v = ctx->GetInputDim("V");
    auto dims_columns = ctx->GetInputDim("Columns");

    PADDLE_ENFORCE_EQ(dims_q.size(), static_cast<size_t>(4),
                      platform::errors::InvalidArgument(
                          "Dimension in query' shapes should be 4."));
    PADDLE_ENFORCE_EQ(dims_k.size(), static_cast<size_t>(4),
                      platform::errors::InvalidArgument(
                          "Dimension in key' shapes should be 4."));
    PADDLE_ENFORCE_EQ(dims_v.size(), static_cast<size_t>(4),
                      platform::errors::InvalidArgument(
                          "Dimension in value' shapes should be 4."));

    auto batch_size = dims_q[0];
    auto num_heads = dims_q[1];
    auto M = dims_q[2];
    auto N = dims_q[3];
    auto sparse_nnz = dims_columns[2];
    ctx->SetOutputDim("Out", {batch_size, num_heads, M, N});
    ctx->SetOutputDim("SparseDotSdd", {batch_size, num_heads, sparse_nnz});
    ctx->SetOutputDim("Softmax", {batch_size, num_heads, sparse_nnz});
    ctx->ShareLoD("Q", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "Q", "K");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class SparseAttentionOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Q"), "Input", "Q", "sparse_attention_grad");
    OP_INOUT_CHECK(ctx->HasInput("K"), "Input", "K", "sparse_attention_grad");
    OP_INOUT_CHECK(ctx->HasInput("V"), "Input", "V", "sparse_attention_grad");
    OP_INOUT_CHECK(ctx->HasInput("Offset"), "Input", "Offset",
                   "sparse_attention_grad");
    OP_INOUT_CHECK(ctx->HasInput("Columns"), "Input", "Columns",
                   "sparse_attention_grad");
    OP_INOUT_CHECK(ctx->HasInput("SparseDotSdd"), "Input", "SparseDotSdd",
                   "sparse_attention_grad");
    OP_INOUT_CHECK(ctx->HasInput("Softmax"), "Input", "Softmax",
                   "sparse_attention_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "sparse_attention_grad");

    auto x_grad_name = framework::GradVarName("Q");
    auto y_grad_name = framework::GradVarName("K");
    auto z_grad_name = framework::GradVarName("V");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("Q"));
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, ctx->GetInputDim("K"));
    }
    if (ctx->HasOutput(z_grad_name)) {
      ctx->SetOutputDim(z_grad_name, ctx->GetInputDim("V"));
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class SparseAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sparse_attention_grad");
    op->SetInput("Q", this->Input("Q"));
    op->SetInput("K", this->Input("K"));
    op->SetInput("V", this->Input("V"));
    op->SetInput("Offset", this->Input("Offset"));
    op->SetInput("Columns", this->Input("Columns"));
    op->SetInput("SparseDotSdd", this->Output("SparseDotSdd"));
    op->SetInput("Softmax", this->Output("Softmax"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Q"), this->InputGrad("Q"));
    op->SetOutput(framework::GradVarName("K"), this->InputGrad("K"));
    op->SetOutput(framework::GradVarName("V"), this->InputGrad("V"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sparse_attention, ops::SparseAttentionOp,
                  ops::SparseAttentionOpMaker,
                  ops::SparseAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::SparseAttentionGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(sparse_attention_grad, ops::SparseAttentionOpGrad);
