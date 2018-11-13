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

#include "paddle/fluid/operators/fused_hash_embedding_seq_pool_op.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class FusedHashEmbeddingSeqPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input X of FusedHashEmbeddingSeqPoolOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("W"),
        "Input W of FusedHashEmbeddingSeqPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output of FusedEmbeddingSeqPoolOp should not be null.");

    // check x dims
    auto x_dims = ctx->GetInputsDim("X");
    size_t win_count = x_dims.size();
    PADDLE_ENFORCE_GE(win_count, 1,
                      "Input tensors' count should be at least one");
    for (auto dim : x_dims) {
      PADDLE_ENFORCE_EQ(dim.size(), 2UL,
                        "The input of hash_op's dimensions must be 2");
    }

    // check tabel dims
    auto table_dims = ctx->GetInputDim("W");
    PADDLE_ENFORCE_EQ(table_dims.size(), 2);

    const std::string& combiner = ctx->Attrs().Get<std::string>("combiner");
    // we only support sum now
    PADDLE_ENFORCE_EQ(combiner, "sum");

    int num_hash = ctx->Attrs().Get<int>("num_hash");
    int64_t last_dim = table_dims[1] * num_hash;

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
      const auto& x_lod = x_var->Get<LoDTensor>().lod();

      // in run time, the LoD of ids must be 1
      PADDLE_ENFORCE(x_lod.size(), 1u, "The LoD level of Input(X) must be 1");
      PADDLE_ENFORCE_GE(x_lod[0].size(), 1u, "The LoD could NOT be empty");

      int64_t batch_size = x_lod[0].size() - 1;

      // in run time, the shape of out should be [batch_size, embedding_size]
      ctx->SetOutputDim("Out", framework::make_ddim({batch_size, last_dim}));
    } else {
      // in compile time, the lod level of ids must be 1
      framework::VarDesc* x_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
      PADDLE_ENFORCE_EQ(x_desc->GetLoDLevel(), 1);

      // in compile time, the shape of output should be [-1, embedding_size]
      ctx->SetOutputDim("Out", framework::make_ddim({-1, last_dim}));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FusedHashEmbeddingSeqPoolOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("X",
             "An input with type int32 or int64 "
             "contains the id sequence to be looked up in W. ")
        .AsDuplicable();
    AddOutput("Out", "The lookup results, which have the same type as W.");
    AddAttr<std::string>("combiner",
                         "(string, default sum) "
                         "A string specifying the reduction op. Currently sum "
                         "are supported, sum computes the weighted sum of the "
                         "embedding results for each row.")
        .SetDefault("sum");
    AddAttr<int64_t>("num_hash",
                     "(int, default 1) "
                     "An integer specifying the hash times")
        .SetDefault(1);
    AddAttr<int64_t>("mod_by",
                     "(int, default 100000)"
                     "An integer specifiying the hash space")
        .SetDefault(100000);
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddComment(R"DOC(
FusedHashEmbeddingSeqPoolOp Operator.

Computes embeddings for the given ids and weights.

This operator is used to perform lookups on the parameter W,
then computes the weighted sum of the lookups results for each row
and concatenated into a dense tensor.

The input X should carry the LoD (Level of Details) information.
And the output will throw the LoD information away.

)DOC");
  }
};

class FusedHashEmbeddingSeqPoolOpGradDescMaker
    : public framework::DefaultGradOpDescMaker<true> {
  using ::paddle::framework::DefaultGradOpDescMaker<
      true>::DefaultGradOpDescMaker;

 protected:
  virtual std::string GradOpType() const {
    return "fused_hash_embedding_seq_pool_grad";
  }
};

class FusedHashEmbeddingSeqPoolOpGrad : public framework::OperatorWithKernel {
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

class FusedHashEmbeddingSeqPoolOpGradVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    auto out_var_name = op_desc.Output(framework::GradVarName("W")).front();
    auto attr = op_desc.GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "fused_hash_embedding_seq_pool_grad op "
              << framework::GradVarName("W") << " is set to SelectedRows";
      block->Var(out_var_name)
          ->SetType(framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "fused_hash_embedding_seq_pool_grad op "
              << framework::GradVarName("W") << " is set to LoDTensor";
      block->Var(out_var_name)->SetType(framework::proto::VarType::LOD_TENSOR);
    }
    block->Var(out_var_name)->SetDataType(block->Var("W")->GetDataType());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_hash_embedding_seq_pool,
                  ops::FusedHashEmbeddingSeqPoolOp,
                  ops::FusedHashEmbeddingSeqPoolOpGradDescMaker,
                  ops::FusedHashEmbeddingSeqPoolOpMaker);
REGISTER_OPERATOR(fused_hash_embedding_seq_pool_grad,
                  ops::FusedHashEmbeddingSeqPoolOpGrad,
                  ops::FusedHashEmbeddingSeqPoolOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(fused_hash_embedding_seq_pool,
                       ops::FusedHashEmbeddingSeqPoolKernel<float>,
                       ops::FusedHashEmbeddingSeqPoolKernel<double>);
REGISTER_OP_CPU_KERNEL(fused_hash_embedding_seq_pool_grad,
                       ops::FusedHashEmbeddingSeqPoolGradKernel<float>,
                       ops::FusedHashEmbeddingSeqPoolGradKernel<double>);
