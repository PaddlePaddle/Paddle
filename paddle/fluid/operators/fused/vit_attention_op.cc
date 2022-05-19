/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace operators {

class VitAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInput("Input"), true,
        platform::errors::InvalidArgument(
            "Input(Input) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of MultiHeadMatMul should not be null."));

    auto dim_input = context->GetInputDim("Input");
    int batch = dim_input[0];
    int seq_len = dim_input[1];
    int hidden_size = dim_input[2] / 3;
    std::vector<int> dims = {batch, seq_len, hidden_size};
    context->SetOutputDim("Out", phi::make_ddim(dims));
  }
};

class VitAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input of VitAttention op");
    AddOutput("Out", "The output of Vitattention op");
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(12);
    AddAttr<float>("scale", "The scale after q*k").SetDefault(1.0f);
    AddComment(R"DOC(
MultiHeadMatMul Operator.

This op is used for optimize multi head calculation in vit model.
Not suggest to use in other case except has same structure as vit_base.

- X: [batch, length, hidden_size * 3]=[batch,length,Q:K:V] => Out: [batch, length, hidden_size]

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(vit_attention, ops::VitAttentionOp,
                             ops::VitAttentionOpMaker);
