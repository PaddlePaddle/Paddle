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

#include "paddle/fluid/operators/merge_ids_op.h"

namespace paddle {
namespace operators {

class MergeIdsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids", "(LoDTensor) the input ids with shape{batch_num, 1}");
    AddInput("X",
             "(LoDTensor) the input tensor with shape{batch_num, N}, N is the "
             "size of embedding table")
        .AsDuplicable();
    AddOutput("Out", "(LoDTensor) The merged outputs of the input tensors.");

    AddComment(R"DOC(
Merge multi LoDTensor's into one according to Ids's shard num.
The values in the input LoDTensor are lookuped from the output of splite_ids_op
Example:
  Input:
    Ids = [1,2,3,4,5,6]
    X0 = [[0.1 0.2]   # 3
          [0.2 0.3]]  # 6
    X1 = [[0.3 0.4]   # 1
          [0.4 0.5]]  # 4
    X2 = [[0.5 0.6]   # 2
          [0.6 0.7]]  # 5

  Output:
    Out = [[0.3 0.4]  # 1
           [0.5 0.6]  # 2
           [0.1 0.2]  # 3
           [0.4 0.5]  # 4
           [0.6 0.7]  # 5
           [0.2 0.3]] # 6
)DOC");
  }
};

class MergeIdsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Ids"), "MergeIdsOp must has input Ids.");
    PADDLE_ENFORCE(ctx->HasInputs("X"), "MergeIdsOp must has input X.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "MergeIdsOp must has output Out.");

    auto ids_var_type = ctx->GetInputsVarType("Ids").front();
    auto ids_dims = ctx->GetInputDim("Ids");
    if (ids_var_type == framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(ids_dims.size(), 2);
      PADDLE_ENFORCE_EQ(ids_dims[1], 1);
    }
    auto x_var_type = ctx->GetInputsVarType("X");
    for (auto &var_type : x_var_type) {
      PADDLE_ENFORCE_EQ(var_type, framework::proto::VarType::LOD_TENSOR,
                        "input X only support lod tensors");
    }
    ctx->ShareLoD("Ids", "Out");
  }

 private:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.MultiInput<framework::Tensor>("X").front()->type()),
        ctx.GetPlace());
  }
};

class MergeIdsOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    auto *input_var = block->Var(op_desc.Input("Ids")[0]);
    for (auto &out_var : op_desc.Output("Out")) {
      block->Var(out_var)->SetType(input_var->GetType());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(merge_ids, ops::MergeIdsOp, ops::MergeIdsOpMaker,
                  ops::MergeIdsOpInferVarType);
REGISTER_OP_CPU_KERNEL(
    merge_ids, ops::MergeIdsOpKernel<paddle::platform::CPUPlace, float>);
