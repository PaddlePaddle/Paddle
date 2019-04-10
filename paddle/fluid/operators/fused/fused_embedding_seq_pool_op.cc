/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_embedding_seq_pool_op.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class FusedEmbeddingSeqPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input W of FusedEmbeddingSeqPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ids"),
                   "Input Ids of FusedEmbeddingSeqPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output of FusedEmbeddingSeqPoolOp should not be null.");

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    const std::string& combiner = ctx->Attrs().Get<std::string>("combiner");

    PADDLE_ENFORCE_EQ(table_dims.size(), 2);
    PADDLE_ENFORCE_GE(ids_dims.size(), 1,
                      "The dim size of the 'Ids' tensor must greater than 1.");
    PADDLE_ENFORCE_EQ(ids_dims[ids_dims.size() - 1], 1,
                      "The last dimension of the 'Ids' tensor must be 1.");
    // we only support sum now
    PADDLE_ENFORCE_EQ(combiner, "sum");

    int64_t last_dim = FusedEmbeddingSeqPoolLastDim(table_dims, ids_dims);
    // in compile time, the lod level of ids must be 1
    framework::VarDesc* ids_desc =
        boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("Ids")[0]);
    PADDLE_ENFORCE_EQ(ids_desc->GetLoDLevel(), 1);

    // in compile time, the shape from Ids -> output
    // should be [-1, 1] -> [-1, embedding_size]
    ctx->SetOutputDim("Out", framework::make_ddim({-1, last_dim}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FusedEmbeddingSeqPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int32 or int64 "
             "contains the ids to be looked up in W. "
             "The last dimension size must be 1.");
    AddOutput("Out", "The lookup results, which have the same type as W.");
    AddAttr<std::string>("combiner",
                         "(string, default sum) "
                         "A string specifying the reduction op. Currently sum "
                         "are supported, sum computes the weighted sum of the "
                         "embedding results for each row.")
        .SetDefault("sum");
    // NOTE(minqiyang): grad_inplace is an temporal attribute,
    // please do NOT set this attribute in python layer.
    AddAttr<bool>("grad_inplace",
                  "(boolean, default false) "
                  "If the grad op reuse the input's variable.")
        .SetDefault(false);
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);
    AddComment(R"DOC(
FusedEmbeddingSeqPool Operator.

Computes embeddings for the given ids and weights.

This operator is used to perform lookups on the parameter W,
then computes the weighted sum of the lookups results for each row
and concatenated into a dense tensor.

The input Ids should carry the LoD (Level of Details) information.
And the output will change the LoD information with input Ids.

)DOC");
  }
};

class FusedEmbeddingSeqPoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto table_dims = ctx->GetInputDim("W");
    ctx->SetOutputDim(framework::GradVarName("W"), table_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FusedEmbeddingSeqPoolOpGradVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto out_var_name = ctx->Output(framework::GradVarName("W")).front();
    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "fused_embedding_seq_pool_grad op "
              << framework::GradVarName("W") << " is set to SelectedRows";
      ctx->SetType(out_var_name, framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "fused_embedding_seq_pool_grad op "
              << framework::GradVarName("W") << " is set to LoDTensor";
      ctx->SetType(out_var_name, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetDataType(out_var_name, ctx->GetDataType(ctx->Input("W")[0]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_embedding_seq_pool, ops::FusedEmbeddingSeqPoolOp,
                  paddle::framework::DefaultGradOpDescMaker<true>,
                  ops::FusedEmbeddingSeqPoolOpMaker);
REGISTER_OPERATOR(fused_embedding_seq_pool_grad,
                  ops::FusedEmbeddingSeqPoolOpGrad,
                  ops::FusedEmbeddingSeqPoolOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(fused_embedding_seq_pool,
                       ops::FusedEmbeddingSeqPoolKernel<float>,
                       ops::FusedEmbeddingSeqPoolKernel<double>);
REGISTER_OP_CPU_KERNEL(fused_embedding_seq_pool_grad,
                       ops::FusedEmbeddingSeqPoolGradKernel<float>,
                       ops::FusedEmbeddingSeqPoolGradKernel<double>);
