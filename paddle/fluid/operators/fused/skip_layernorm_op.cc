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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace operators {

class SkipLayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(context->HasInput("Y"), true,
                      platform::errors::InvalidArgument(
                          "Input(Y) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasInput("Scale"), true,
        platform::errors::InvalidArgument(
            "Input(Scale) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasInput("Bias"), true,
        platform::errors::InvalidArgument(
            "Input(Bias) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of MultiHeadMatMul should not be null."));

    auto dim_input = context->GetInputDim("X");
    context->SetOutputDim("Out", dim_input);
    context->ShareLoD("X", "Out");
  }
};

class SkipLayerNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The X input of SkipLayerNorm op");
    AddInput("Y", "The Y input of SkipLayerNorm op");
    AddInput("Scale", "The scale input of SkipLayerNorm op");
    AddInput("Bias", "The bias input of SkipLayerNorm op");
    AddOutput("Out", "The output of SkipLayerNorm op");
    AddAttr<float>("epsilon",
                   "param epsilon of layer_norm op in "
                   "skip_layernorm_fuse_pass");
    AddAttr<int>("begin_norm_axis",
                 "param begin_norm_axis of "
                 "layer_norm op in skip_layernorm_fuse_pass");
    AddComment(R"DOC(
SkipLayerNorm Operator.

This op is used for skip_layernorm_fuse_pass, which fuse op pattern as followed.

     |           |                            |            |
 other_op1   other_op2                    other_op1    other_op2
     |           |              fuse           \          /
     |------elementwise_add      ->           skip_layernorm
                 |                                   |
             layer_norm                          other_op3
                 |                                   |
             other_op3
                 |

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(skip_layernorm, ops::SkipLayerNormOp,
                             ops::SkipLayerNormOpMaker);
