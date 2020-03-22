//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/push_dense_op.h"

namespace paddle {
namespace operators {

class PushDenseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("DenseGrads").size(), 1UL,
                      "Input(DenseGrads) of PushDenseOp should not be null.");
    PADDLE_ENFORCE_GE(ctx->Inputs("Ids").size(), 1UL,
                      "Input(Ids) of PushDenseOp should not be null.");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class PushDenseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("DenseGrads", "The dense gradient tensors").AsDuplicable();
    AddInput("Ids", "the tensor to get batch size").AsDuplicable();
    AddAttr<int>("TableId", "(int, the table id of this embedding").SetDefault(-1);;
    AddAttr<float>("ScaleDataNorm", "(float, scale data norm gradient").SetDefault(-1.0f);
    AddAttr<std::vector<std::string>>("InputNames", "(vector, slot names").SetDefault(std::vector<std::string>());
    AddComment(R"DOC(
Push Dense Operator.

push dense gradients to PSLib's Parameter Server.

The input gradients can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(push_dense, ops::PushDenseOp,
                  ops::PushDenseOpMaker,
                  paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
                 paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(push_dense, ops::PushDenseCPUKernel<float>)
