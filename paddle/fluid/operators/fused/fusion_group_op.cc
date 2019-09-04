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

#include "paddle/fluid/operators/fused/fusion_group_op.h"

namespace paddle {
namespace operators {

class FusionGroupOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      "The number of inputs should be no less than 1.");
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      "The number of outputs should be no less than 1.");

    const size_t num_ins = ctx->Inputs("X").size();
    const size_t num_outs = ctx->Outputs("Out").size();

    int type = ctx->Attrs().Get<int>("type");
    PADDLE_ENFORCE_EQ(type, 0UL,
                      "Only support fusion of elementwise operations.");

    std::vector<framework::DDim> x_dims = ctx->GetInputsDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), num_ins);

    if (type == 0) {
      for (size_t i = 1; i < num_ins; ++i) {
        PADDLE_ENFORCE_EQ(x_dims[0], x_dims[i],
                          "All the inputs' dims should be the same.");
      }
      std::vector<framework::DDim> out_dims;
      for (size_t j = 0; j < num_outs; ++j) {
        out_dims.push_back(x_dims[0]);
      }
      ctx->SetOutputsDim("Out", out_dims);
    }

    // Only lod of X[0] would be shared with Out.
    for (size_t j = 0; j < num_outs; ++j) {
      ctx->ShareLoD("X", /*->*/ "Out", 0, j);
    }
  }
};

class FusionGroupOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The inputs of fusion_group op.").AsDuplicable();
    AddOutput("Out", "The outputs of fusion_group op.").AsDuplicable();
    AddAttr<int>("type", "Fusion type.").SetDefault(0);
    AddAttr<std::string>("func_name", "Name of the generated codes.")
        .SetDefault("");
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_group, ops::FusionGroupOp, ops::FusionGroupOpMaker);
