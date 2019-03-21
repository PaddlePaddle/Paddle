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

#include "paddle/fluid/operators/sequence_ops/sequence_pyramid_embedding_op.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class SequencePyramidEmbeddingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->IsRuntime()) {
      return;
    }
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input W of SequencePyramidEmbeddingOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ids"),
                   "Input Ids of SequencePyramidEmbeddingOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output of SequencePyramidEmbeddingOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("HashIds"),
                   "Output of SequencePyramidEmbeddingOp should not be null.");

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    const int rand_len = ctx->Attrs().Get<int>("rand_len");
    const int num_hash = ctx->Attrs().Get<int>("num_hash");

    PADDLE_ENFORCE_EQ(table_dims.size(), 2);
    PADDLE_ENFORCE_GE(ids_dims.size(), 1,
                      "The dim size of the 'Ids' tensor must greater than 1.");
    PADDLE_ENFORCE_EQ(ids_dims[ids_dims.size() - 1], 1,
                      "The last dimension of the 'Ids' tensor must be 1.");
    // in compile time, the lod level of ids must be 1
    framework::VarDesc* ids_desc =
        boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("Ids")[0]);
    PADDLE_ENFORCE_EQ(ids_desc->GetLoDLevel(), 1);

    // in compile time, the shape from Ids -> output
    // should be [-1, 1] -> [-1, embedding_size]
    ctx->SetOutputDim("Out", framework::make_ddim({-1, rand_len * num_hash}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SequencePyramidEmbeddingOpMaker : public framework::OpProtoAndCheckerMaker {
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
    AddOutput("HashIds", "The sequence pyramid hash ids.");
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddAttr<int>("rand_len",
                         "(int, default 16) "
                         "Random length is the size of vertor obtained from "
                         "embedding tendor W, while every id of Ids gets a vector"
                         " from W")
        .SetDefault(16);
    AddAttr<int>("min_win_size", "(int) The enumerate sequence's min window size.")
        .AddCustomChecker([](const int& min_win_size) {
          PADDLE_ENFORCE(min_win_size >= 2,
                         "The window size should be not less than 2.");
        });
    AddAttr<int>("max_win_size", "(int) The enumerate sequence's max window size.")
        .AddCustomChecker([](const int& max_win_size) {
          PADDLE_ENFORCE(max_win_size >= 2,
                         "The window size should be not less than 2.");
        });
    AddAttr<int>("num_hash", "").SetDefault(1);
    AddAttr<int>("mod_by", "").SetDefault(100000);
    AddAttr<float>("dropout_rate", "").SetDefault(0.0);
    AddAttr<bool>("fix_seed", "").SetDefault(false);
    AddAttr<int>("seed", "").SetDefault(123);
    AddAttr<int>("white_list_len", "").SetDefault(0);
    AddAttr<int>("black_list_len", "").SetDefault(0);
    AddInput("BlackFilter", "").AsDispensable();
    AddInput("Filter", "").AsDispensable();

    AddComment(R"DOC(
SequencePyramidEmbedding Operator.

Computes embeddings for the given ids and weights.

This operator is used to perform lookups on the parameter W,
then computes the weighted sum of the lookups results for each row
and concatenated into a dense tensor.

The input Ids should carry the LoD (Level of Details) information.
And the output will change the LoD information with input Ids.

)DOC");
  }
};

class SequencePyramidEmbeddingOpGradDescMaker
    : public framework::DefaultGradOpDescMaker<true> {
  using ::paddle::framework::DefaultGradOpDescMaker<
      true>::DefaultGradOpDescMaker;

 protected:
  virtual std::string GradOpType() const {
    return "sequence_pyramid_embedding_grad";
  }
};

class SequencePyramidEmbeddingOpGrad : public framework::OperatorWithKernel {
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

class SequencePyramidEmbeddingOpGradVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    auto out_var_name = op_desc.Output(framework::GradVarName("W")).front();
    auto attr = op_desc.GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "sequence_pyramid_embedding_grad op "
              << framework::GradVarName("W") << " is set to SelectedRows";
      block->Var(out_var_name)
          ->SetType(framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "sequence_pyramid_embedding_grad op "
              << framework::GradVarName("W") << " is set to LoDTensor";
      block->Var(out_var_name)->SetType(framework::proto::VarType::LOD_TENSOR);
    }
    block->Var(out_var_name)->SetDataType(block->Var("W")->GetDataType());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_pyramid_embedding, ops::SequencePyramidEmbeddingOp,
                  ops::SequencePyramidEmbeddingOpGradDescMaker,
                  ops::SequencePyramidEmbeddingOpMaker);
REGISTER_OPERATOR(sequence_pyramid_embedding_grad,
                  ops::SequencePyramidEmbeddingOpGrad,
                  ops::SequencePyramidEmbeddingOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(sequence_pyramid_embedding,
                       ops::SequencePyramidEmbeddingKernel<float>,
                       ops::SequencePyramidEmbeddingKernel<double>);
REGISTER_OP_CPU_KERNEL(sequence_pyramid_embedding_grad,
                       ops::SequencePyramidEmbeddingGradKernel<float>,
                       ops::SequencePyramidEmbeddingGradKernel<double>);
