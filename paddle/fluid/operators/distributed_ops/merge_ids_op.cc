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

#include "paddle/fluid/operators/distributed_ops/merge_ids_op.h"

namespace paddle {
namespace operators {

class MergeIdsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids", "(LoDTensor) the input ids with shape{batch_num, 1}")
        .AsDuplicable();
    AddInput("Rows", "(LoDTensor) the input ids with shape{row_size, 1}, ")
        .AsDuplicable();
    AddInput("X",
             "(LoDTensors) multi input tensor with shape{Rows, N}, N is the "
             "size of embedding table")
        .AsDuplicable();
    AddOutput("Out", "(LoDTensor) The merged outputs of the input tensors.")
        .AsDuplicable();

    AddComment(R"DOC(
Merge multi LoDTensor's into one according to Ids's shard num.


split_ids_op -> prefetch_op -> merge_ids_op


merge_ids_op should be used after split_ids_op and prefetch_op, split_ids_op
 will split input Ids into multiple tensors according to Id's shard number.
prefetch_op will send them to parameter server to prefetch embedding value
back. During split, the order of ids is disordered. In merge_ids_op we use
the original Ids to restore the order of the fetched embedding value and
 also pass the lod information to the merged output.


Example:

    Ids = [1,2,3,4,5,6] # 3 shared

split_ids_op ->

    Id0 = [3, 6] # id % 3 == 0
    Id1 = [1, 4] # id % 3 == 1
    Id2 = [2, 5] # id % 3 == 2

prefetch_op ->

    X0 = [[0.3 0.3]   # 3
          [0.6 0.6]]  # 6
    X1 = [[0.1 0.1]   # 1
          [0.4 0.4]]  # 4
    X2 = [[0.2 0.2]   # 2
          [0.5 0.5]]  # 5

merge_ids_op ->

    Out = [[0.1 0.1]  # 1
           [0.2 0.2]  # 2
           [0.3 0.3]  # 3
           [0.4 0.4]  # 4
           [0.5 0.5]  # 5
           [0.6 0.6]] # 6
)DOC");
  }
};

class MergeIdsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("Ids"), "Input", "Ids", "MergeIds");
    OP_INOUT_CHECK(ctx->HasInputs("Rows"), "Input", "Rows", "MergeIds");
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "MergeIds");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "MergeIds");

    auto ids_var_type = ctx->GetInputsVarType("Ids").front();
    auto ids_dims = ctx->GetInputsDim("Ids");
    if (ids_var_type == framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          ids_dims[0].size(), 2,
          platform::errors::InvalidArgument(
              "the ids size must be 2, but received %d", ids_dims[0].size()));
      PADDLE_ENFORCE_EQ(
          ids_dims[0][1], 1,
          platform::errors::InvalidArgument(
              "the ids dim must be 1, but received %d", ids_dims[0][1]));
    }
    auto x_var_type = ctx->GetInputsVarType("X");
    for (auto &var_type : x_var_type) {
      PADDLE_ENFORCE_EQ(var_type, framework::proto::VarType::LOD_TENSOR,
                        platform::errors::InvalidArgument(
                            "input X only support lod tensors"));
    }
    ctx->ShareLoD("Ids", "Out");
  }

 private:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class MergeIdsOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto input_type = ctx->GetInputType("Ids");
    ctx->SetOutputType("Out", input_type, framework::ALL_ELEMENTS);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(merge_ids, ops::MergeIdsOp, ops::MergeIdsOpMaker,
                  ops::MergeIdsOpInferVarType);
REGISTER_OP_CPU_KERNEL(
    merge_ids, ops::MergeIdsOpKernel<paddle::platform::CPUPlace, float>);
