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

#include "paddle/fluid/operators/split_ids_op.h"

namespace paddle {
namespace operators {

class SplitIdsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids", "(LoDTensor) the input ids with shape{batch_num, 1}");
    AddOutput("Out", "(LoDTensor) The outputs of the input Ids.")
        .AsDuplicable();

    AddComment(R"DOC(
Split a LoDTensor of Ids into multi LoDTensors, the number is pserver's number
Example:
  Input:
    X = [1,2,3,4,5,6]

  Out(3 output):
    out0 = [3, 6]
    out1 = [1, 4]
    out2 = [2, 5]
)DOC");
  }
};

class SplitIdsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Ids"), "SplitIdsOp must has input Ids.");
    PADDLE_ENFORCE(ctx->HasOutputs("Out"), "SplitIdsOp must has output Out.");

    auto ids_var_type = ctx->GetInputsVarType("Ids").front();
    auto ids_dims = ctx->GetInputDim("Ids");
    if (ids_var_type == framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(ids_dims.size(), 2);
      PADDLE_ENFORCE_EQ(ids_dims[1], 1);
    }
  }
};

class SplitIdsOpInferVarType : public framework::VarTypeInference {
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
REGISTER_OPERATOR(split_ids, ops::SplitIdsOp, ops::SplitIdsOpMaker,
                  ops::SplitIdsOpInferVarType);
REGISTER_OP_CPU_KERNEL(
    split_ids, ops::SplitIdsOpKernel<paddle::platform::CPUPlace, int64_t>,
    ops::SplitIdsOpKernel<paddle::platform::CPUPlace, float>);
