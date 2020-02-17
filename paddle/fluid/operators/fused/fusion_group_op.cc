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
    const size_t num_ins = ctx->Inputs("Inputs").size();
    const size_t num_outs = ctx->Outputs("Outs").size();

    PADDLE_ENFORCE_GE(
        num_ins, 1UL,
        platform::errors::InvalidArgument(
            "Expected the number of inputs >= 1. Received %d.", num_ins));
    PADDLE_ENFORCE_GE(
        num_outs, 1UL,
        platform::errors::InvalidArgument(
            "Expected the number of outputs >= 1. Recived %d.", num_outs));

    int type = ctx->Attrs().Get<int>("type");
    PADDLE_ENFORCE_EQ(type, 0UL,
                      platform::errors::InvalidArgument(
                          "Only support fusion of elementwise operations."));

    std::vector<framework::DDim> x_dims = ctx->GetInputsDim("Inputs");
    if (type == 0) {
      for (size_t i = 1; i < num_ins; ++i) {
        PADDLE_ENFORCE_EQ(x_dims[0], x_dims[i],
                          platform::errors::InvalidArgument(
                              "All the inputs' dims should be the same."));
      }
      std::vector<framework::DDim> out_dims;
      for (size_t j = 0; j < num_outs; ++j) {
        out_dims.push_back(x_dims[0]);
      }
      ctx->SetOutputsDim("Outs", out_dims);
    }

    // Only lod of Inputs[0] would be shared with Outs.
    for (size_t j = 0; j < num_outs; ++j) {
      ctx->ShareLoD("Inputs", /*->*/ "Outs", 0, j);
    }
  }
};

class FusionGroupOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Inputs",
             "(std::vector<LoDTensor>) The inputs of fusion_group op.")
        .AsDuplicable();
    AddOutput("Outs",
              "(std::vector<LoDTensor>) The outputs of fusion_group op.")
        .AsDuplicable();
    AddAttr<int>("type", "Fusion type.").SetDefault(0);
    AddAttr<std::string>("func_name", "Name of the generated functions.")
        .SetDefault("");
    AddComment(R"DOC(
fusion_group Operator.

It is used to execute a generated CUDA kernel which fuse the computation of
multiple operators into one. It supports several types:
0, fused computation of elementwise operations in which all the dims of inputs
    and outputs should be exactly the same.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_group, ops::FusionGroupOp, ops::FusionGroupOpMaker);
